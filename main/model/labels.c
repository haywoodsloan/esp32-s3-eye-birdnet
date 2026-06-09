/**
 * @file labels.c
 * @brief Active species label table.
 *
 * The table is dynamic: model_registry loads a per-model labels file from SD
 * via birdnet_labels_load_file(). The built-in default list is used for the
 * mock classifier and as a fallback when no SD labels are available.
 *
 * Written only from the inference task + boot (single writer), so no locking.
 */
#include "labels.h"
#include "birdnet_config.h"

#include <stdio.h>
#include <string.h>

#include "esp_log.h"

#define LBL_NAME_LEN 48
#define LBL_SCI_LEN  64

static const char *TAG = "labels";

static char s_name[BIRDNET_NUM_CLASSES_MAX][LBL_NAME_LEN];
static char s_sci[BIRDNET_NUM_CLASSES_MAX][LBL_SCI_LEN];

int birdnet_num_labels = 0;

/* Built-in default species (used by the mock classifier / fallback). */
static const char *const k_default[][2] = {
    { "American Robin",         "Turdus migratorius"    },
    { "Northern Cardinal",      "Cardinalis cardinalis" },
    { "Blue Jay",               "Cyanocitta cristata"   },
    { "Black-capped Chickadee", "Poecile atricapillus"  },
    { "House Sparrow",          "Passer domesticus"     },
    { "American Goldfinch",     "Spinus tristis"        },
    { "Mourning Dove",          "Zenaida macroura"      },
    { "Red-winged Blackbird",   "Agelaius phoeniceus"   },
    { "Song Sparrow",           "Melospiza melodia"     },
    { "Tufted Titmouse",        "Baeolophus bicolor"    },
    { "Downy Woodpecker",       "Dryobates pubescens"   },
    { "European Starling",      "Sturnus vulgaris"      },
};

const char *birdnet_label_name(int index)
{
    if (index < 0 || index >= birdnet_num_labels) {
        return "?";
    }
    return s_name[index];
}

const char *birdnet_label_sci(int index)
{
    if (index < 0 || index >= birdnet_num_labels) {
        return "?";
    }
    return s_sci[index];
}

void birdnet_labels_reset_default(void)
{
    int n = (int)(sizeof(k_default) / sizeof(k_default[0]));
    if (n > BIRDNET_NUM_CLASSES_MAX) n = BIRDNET_NUM_CLASSES_MAX;
    for (int i = 0; i < n; ++i) {
        strncpy(s_name[i], k_default[i][0], LBL_NAME_LEN - 1);
        s_name[i][LBL_NAME_LEN - 1] = '\0';
        strncpy(s_sci[i], k_default[i][1], LBL_SCI_LEN - 1);
        s_sci[i][LBL_SCI_LEN - 1] = '\0';
    }
    birdnet_num_labels = n;
}

/* Trim trailing CR/LF/space/tab in place. */
static void rstrip(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' ||
                     s[n - 1] == ' '  || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
}

esp_err_t birdnet_labels_load_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        ESP_LOGE(TAG, "cannot open labels: %s", path);
        return ESP_ERR_NOT_FOUND;
    }

    char line[LBL_NAME_LEN + LBL_SCI_LEN + 4];
    int count = 0;
    while (count < BIRDNET_NUM_CLASSES_MAX && fgets(line, sizeof(line), f) != NULL) {
        rstrip(line);
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }

        char *sep = strpbrk(line, "|\t");
        const char *common = line;
        const char *sci = line;
        if (sep != NULL) {
            *sep = '\0';
            common = line;
            sci = sep + 1;
            while (*sci == ' ') sci++;
        }

        snprintf(s_name[count], LBL_NAME_LEN, "%s", common);
        snprintf(s_sci[count], LBL_SCI_LEN, "%s", sci);
        count++;
    }
    fclose(f);

    if (count == 0) {
        ESP_LOGE(TAG, "no labels parsed from %s", path);
        return ESP_FAIL;
    }

    birdnet_num_labels = count;
    ESP_LOGI(TAG, "loaded %d labels from %s", count, path);
    return ESP_OK;
}
