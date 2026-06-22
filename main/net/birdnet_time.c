/**
 * @file birdnet_time.c
 * @brief WiFi STA + SNTP time sync with NVS persistence (see birdnet_time.h).
 */
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_wifi_default.h"
#include "esp_netif_sntp.h"
#include "esp_sntp.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "birdnet_config.h"
#include "birdnet_time.h"

#if BIRDNET_USE_PRIOR

/* These come from the "BirdNET detector" menuconfig entry (Kconfig.projbuild).
 * Fall back to empty so a stale sdkconfig (pre-Kconfig) still compiles; the
 * build regenerates sdkconfig from Kconfig and fills these in. */
#ifndef CONFIG_BIRDNET_WIFI_SSID
#define CONFIG_BIRDNET_WIFI_SSID ""
#endif
#ifndef CONFIG_BIRDNET_WIFI_PASSWORD
#define CONFIG_BIRDNET_WIFI_PASSWORD ""
#endif

static const char *TAG = "birdtime";

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1
#define WIFI_MAX_RETRY      8

/* Sanity floor for a "real" epoch (2023-11-14); anything earlier means the
 * clock was never actually set, so we treat the time as unknown. */
#define EPOCH_SANE_MIN      1700000000LL

#define NVS_NAMESPACE       "birdnet"
#define NVS_KEY_EPOCH       "epoch"

static EventGroupHandle_t s_wifi_events;
static volatile bool      s_time_valid = false;
static int                s_retry = 0;

/* ---- NVS persistence ---------------------------------------------------- */
static void persist_epoch(void)
{
    time_t now = time(NULL);
    if (now < EPOCH_SANE_MIN) {
        return;
    }
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) {
        return;
    }
    if (nvs_set_i64(h, NVS_KEY_EPOCH, (int64_t)now) == ESP_OK) {
        nvs_commit(h);
    }
    nvs_close(h);
}

/* Restore the last known epoch so the week is approximately right even before
 * (or without) a WiFi sync. It will be stale by the powered-off duration, which
 * barely matters at week granularity, and SNTP corrects it once connected. */
static void restore_epoch(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) {
        return;   /* no stored time yet */
    }
    int64_t saved = 0;
    if (nvs_get_i64(h, NVS_KEY_EPOCH, &saved) == ESP_OK && saved > EPOCH_SANE_MIN) {
        struct timeval tv = { .tv_sec = (time_t)saved, .tv_usec = 0 };
        settimeofday(&tv, NULL);
        s_time_valid = true;
        ESP_LOGI(TAG, "restored approximate time from NVS (stale until SNTP sync)");
    }
    nvs_close(h);
}

/* ---- WiFi --------------------------------------------------------------- */
static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg; (void)data;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry < WIFI_MAX_RETRY) {
            esp_wifi_connect();
            ESP_LOGW(TAG, "WiFi disconnected, retry %d/%d", ++s_retry, WIFI_MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_events, WIFI_FAIL_BIT);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_retry = 0;
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_connect(void)
{
    s_wifi_events = xEventGroupCreate();
    if (s_wifi_events == NULL) {
        return ESP_ERR_NO_MEM;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    esp_err_t loop_err = esp_event_loop_create_default();
    if (loop_err != ESP_OK && loop_err != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(loop_err);
    }
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &on_wifi_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &on_wifi_event, NULL, NULL));

    wifi_config_t wc = { 0 };
    strlcpy((char *)wc.sta.ssid, CONFIG_BIRDNET_WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, CONFIG_BIRDNET_WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.pmf_cfg.capable  = true;   /* allow, but don't require, WPA3/PMF */
    wc.sta.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "connecting to WiFi \"%s\"...", CONFIG_BIRDNET_WIFI_SSID);
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_events, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));

    return (bits & WIFI_CONNECTED_BIT) ? ESP_OK : ESP_FAIL;
}

/* ---- SNTP --------------------------------------------------------------- */
static esp_err_t sntp_sync(void)
{
    esp_sntp_config_t cfg = ESP_NETIF_SNTP_DEFAULT_CONFIG("pool.ntp.org");
    esp_err_t err = esp_netif_sntp_init(&cfg);   /* starts periodic auto-refresh */
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }
    err = esp_netif_sntp_sync_wait(pdMS_TO_TICKS(15000));
    if (err == ESP_OK) {
        s_time_valid = true;
    }
    return err;
}

/* ---- Background task ----------------------------------------------------- */
static void time_task(void *arg)
{
    (void)arg;

    /* An approximate clock from a previous boot, available immediately. */
    restore_epoch();

    if (strlen(CONFIG_BIRDNET_WIFI_SSID) == 0) {
        ESP_LOGW(TAG, "no WiFi SSID configured (menuconfig -> BirdNET detector); "
                      "seasonal prior runs on stored/unknown time only");
    } else if (wifi_connect() != ESP_OK) {
        ESP_LOGW(TAG, "WiFi connect failed; using stored/unknown time for now");
    } else if (sntp_sync() != ESP_OK) {
        ESP_LOGW(TAG, "SNTP sync timed out; using stored/unknown time for now");
    } else {
        time_t now = time(NULL);
        struct tm tm_utc;
        gmtime_r(&now, &tm_utc);
        char buf[32];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", &tm_utc);
        ESP_LOGI(TAG, "time synced: %s (prior week index %d)",
                 buf, birdnet_time_prior_week());
        persist_epoch();
    }

    /* Keep the stored epoch moving forward and let SNTP's own periodic refresh
     * (started by esp_netif_sntp_init) keep the clock honest. */
    const TickType_t period =
        pdMS_TO_TICKS((uint32_t)BIRDNET_TIME_RESYNC_HOURS * 3600U * 1000U);
    for (;;) {
        vTaskDelay(period);
        if (s_time_valid) {
            persist_epoch();
        }
    }
}

/* ---- Public API --------------------------------------------------------- */
esp_err_t birdnet_time_start(void)
{
    BaseType_t ok = xTaskCreatePinnedToCore(time_task, "birdtime", 4096, NULL, 3, NULL, 0);
    return (ok == pdPASS) ? ESP_OK : ESP_FAIL;
}

int birdnet_time_prior_week(void)
{
    if (!s_time_valid) {
        return -1;
    }
    time_t now = time(NULL);
    if (now < EPOCH_SANE_MIN) {
        return -1;
    }
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);

    /* BirdNET's 4-weeks-per-month convention: week-in-month 0..3 (days 1-7,
     * 8-14, 15-21, 22-end), giving a 0..47 row index into the prior table. */
    int week_in_month = (tm_utc.tm_mday - 1) / 7;
    if (week_in_month > 3) {
        week_in_month = 3;
    }
    return tm_utc.tm_mon * 4 + week_in_month;   /* tm_mon 0..11 -> 0..47 */
}

bool birdnet_time_valid(void)
{
    return s_time_valid;
}

#endif /* BIRDNET_USE_PRIOR */
