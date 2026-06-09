/**
 * @file model_registry.c
 * @brief SD-backed registry of swappable BirdNET expert models (Option B).
 *
 * The manifest is a simple line-based text file (no JSON dependency, since
 * cJSON was removed from ESP-IDF core in v6.0). Each non-comment line is:
 *
 *   Name | model.tflite | labels.txt | default
 *
 * - fields are separated by '|' and trimmed of surrounding whitespace;
 * - the 4th field is optional; the literal "default" (any case) marks the
 *   boot model (otherwise the first entry is the default);
 * - the labels field may be empty to keep the previously loaded labels;
 * - blank lines and lines starting with '#' are ignored.
 */
#include "model_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "birdnet_config.h"
#include "birdnet_model.h"
#include "labels.h"

#include "bsp/esp32_s3_eye.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "model_registry";

#define MODELS_DIR     BSP_SD_MOUNT_POINT BIRDNET_MODELS_SUBDIR
#define MANIFEST_PATH  MODELS_DIR "/" BIRDNET_MANIFEST_NAME
#define NAME_LEN       32
#define FILE_LEN       64

typedef struct {
    char name[NAME_LEN];
    char model_file[FILE_LEN];
    char labels_file[FILE_LEN];
} model_entry_t;

static model_entry_t s_models[BIRDNET_MAX_MODELS];
static int s_count = 0;
static int s_default = 0;
static int s_active = -1;
static volatile int s_pending = -1;

static uint8_t *s_model_buf = NULL;   /* resident, 16-byte aligned PSRAM */

/* Trim leading/trailing whitespace; returns pointer into the same buffer. */
static char *trim(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' '  || s[n - 1] == '\t' ||
                     s[n - 1] == '\r' || s[n - 1] == '\n')) {
        s[--n] = '\0';
    }
    return s;
}

/* Copy a trimmed token into dst (NUL-terminated, bounded). */
static void copy_field(char *dst, size_t dstsz, char *src)
{
    char *t = trim(src);
    snprintf(dst, dstsz, "%s", t);
}

static esp_err_t parse_manifest(void)
{
    FILE *f = fopen(MANIFEST_PATH, "r");
    if (f == NULL) {
        ESP_LOGE(TAG, "cannot open manifest: %s", MANIFEST_PATH);
        return ESP_ERR_NOT_FOUND;
    }

    s_count = 0;
    s_default = 0;
    char line[FILE_LEN * 2 + NAME_LEN + 16];

    while (s_count < BIRDNET_MAX_MODELS && fgets(line, sizeof(line), f) != NULL) {
        char *p = trim(line);
        if (p[0] == '\0' || p[0] == '#') {
            continue;
        }

        /* Split on '|' into up to 4 fields. */
        char *fields[4] = { p, NULL, NULL, NULL };
        int nf = 1;
        for (char *c = p; *c != '\0' && nf < 4; ++c) {
            if (*c == '|') {
                *c = '\0';
                fields[nf++] = c + 1;
            }
        }

        if (nf < 2) {
            ESP_LOGW(TAG, "skipping malformed manifest line: %s", p);
            continue;
        }

        model_entry_t *e = &s_models[s_count];
        copy_field(e->name,        NAME_LEN, fields[0]);
        copy_field(e->model_file,  FILE_LEN, fields[1]);
        if (nf >= 3) {
            copy_field(e->labels_file, FILE_LEN, fields[2]);
        } else {
            e->labels_file[0] = '\0';
        }

        if (e->model_file[0] == '\0') {
            continue;  /* no model file -> skip */
        }
        if (e->name[0] == '\0') {
            snprintf(e->name, NAME_LEN, "%s", e->model_file);
        }
        if (nf >= 4) {
            char *flag = trim(fields[3]);
            if (strcasecmp(flag, "default") == 0 || strcmp(flag, "*") == 0) {
                s_default = s_count;
            }
        }
        s_count++;
    }
    fclose(f);

    if (s_count == 0) {
        ESP_LOGE(TAG, "manifest listed no usable models");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "manifest: %d model(s), default=%d (%s)",
             s_count, s_default, s_models[s_default].name);
    return ESP_OK;
}

esp_err_t model_registry_init(void)
{
    esp_err_t err = bsp_sdcard_mount();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "SD mount failed: %s", esp_err_to_name(err));
        return err;
    }

    err = parse_manifest();
    if (err != ESP_OK) {
        return err;
    }

    s_model_buf = (uint8_t *)heap_caps_aligned_alloc(
        16, BIRDNET_MODEL_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_model_buf == NULL) {
        ESP_LOGE(TAG, "failed to allocate %d byte model buffer",
                 BIRDNET_MODEL_MAX_BYTES);
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

int model_registry_count(void) { return s_count; }
int model_registry_default_index(void) { return s_default; }

const char *model_registry_name(int index)
{
    if (index < 0 || index >= s_count) {
        return "";
    }
    return s_models[index].name;
}

const char *model_registry_active_name(void)
{
    if (s_active < 0 || s_active >= s_count) {
        return "";
    }
    return s_models[s_active].name;
}

esp_err_t model_registry_apply(int index)
{
    if (index < 0 || index >= s_count || s_model_buf == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    const model_entry_t *e = &s_models[index];

    char path[160];
    snprintf(path, sizeof(path), "%s/%s", MODELS_DIR, e->model_file);

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "cannot open model: %s", path);
        return ESP_ERR_NOT_FOUND;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > BIRDNET_MODEL_MAX_BYTES) {
        ESP_LOGE(TAG, "model size %ld out of range (max %d)", sz, BIRDNET_MODEL_MAX_BYTES);
        fclose(f);
        return ESP_FAIL;
    }
    size_t rd = fread(s_model_buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        ESP_LOGE(TAG, "short read on model (%u/%ld)", (unsigned)rd, sz);
        return ESP_FAIL;
    }

    esp_err_t err = birdnet_model_set(s_model_buf, (size_t)sz);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "birdnet_model_set failed for %s", e->name);
        return err;
    }

    /* Labels: prefer the model's labels file; fall back to keeping current. */
    if (e->labels_file[0] != '\0') {
        char lpath[160];
        snprintf(lpath, sizeof(lpath), "%s/%s", MODELS_DIR, e->labels_file);
        if (birdnet_labels_load_file(lpath) != ESP_OK) {
            ESP_LOGW(TAG, "labels load failed; keeping previous labels");
        }
    }

    s_active = index;
    ESP_LOGI(TAG, "active model: %s (%ld bytes), free PSRAM=%u KB",
             e->name, sz,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    return ESP_OK;
}

void model_registry_request(int index)
{
    if (index >= 0 && index < s_count) {
        s_pending = index;
    }
}

void model_registry_request_next(void)
{
    if (s_count <= 0) {
        return;
    }
    int base = (s_active >= 0) ? s_active : s_default;
    s_pending = (base + 1) % s_count;
}

int model_registry_take_pending(void)
{
    int p = s_pending;
    if (p >= 0) {
        s_pending = -1;
    }
    return p;
}
