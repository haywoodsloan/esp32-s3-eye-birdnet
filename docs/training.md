# Training a model for ESP32-S3-EYE BirdNET

This firmware runs a [birdnet-stm32](https://github.com/birdnet-team/birdnet-stm32)
**DS-CNN** with the **`hybrid`** audio frontend. The model is self-contained: it
takes a linear `|STFT|` magnitude spectrogram and internally does the mel
projection, normalization, PWL magnitude scaling, DS-CNN body, and sigmoid head.
The device only computes the STFT.

> You run training on a Linux/macOS/Windows machine with Python + TensorFlow
> (a GPU helps a lot). The firmware author can write/adjust these scripts, but
> the training run, conversion, and on-board flashing are yours to execute.

---

## The contract (must match the firmware)

These values are fixed in [../main/birdnet_config.h](../main/birdnet_config.h).
Train with exactly these or the on-device features won't match the model.

| Parameter | Value | birdnet-stm32 arg |
|---|---|---|
| Sample rate | 24000 Hz | `--sample_rate 24000` |
| FFT length | 512 | `--fft_length 512` |
| Spectrogram width | 256 frames | `--spec_width 256` |
| Chunk duration | 3 s | `--chunk_duration 3` |
| Mel bands (in-model) | 64 | `--num_mels 64` |
| Frontend | hybrid | `--audio_frontend hybrid` |
| Magnitude scaling | pwl | `--mag_scale pwl` |
| Width multiplier | 0.5 (recommended) | `--alpha 0.5` |

Model input: `[1, 257, 256, 1]` (float32 or int8). Output: `[1, num_classes]`
sigmoid scores.

> **Why `--alpha 0.5`?** The ESP32-S3 has no NPU (unlike the STM32N6 this tool
> targets). A smaller width keeps the tensor arena and per-inference time
> reasonable. Start at 0.5; try 0.75/1.0 only if accuracy needs it and the arena
> still fits `BIRDNET_TFLITE_ARENA_BYTES`.

---

## 1. Set up birdnet-stm32

```bash
git clone https://github.com/birdnet-team/birdnet-stm32.git
cd birdnet-stm32
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## 2. Get training data

Pick your local species in [species.txt](species.txt) (scientific names).

- **Birds** from xeno-canto (needs a free account/API key):
  ```bash
  xc-download --api-key <KEY> --species-file species.txt --n-recs 200
  ```
- **Negatives:** put non-bird audio in folders named `noise`, `silence`,
  `background`, or `other` under `data/train/` — birdnet-stm32 gives these
  all-zero labels so the model learns to stay quiet.
- Optionally add your own field recordings (organized as `data/train/<species>/`).

## 3. Train

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --audio_frontend hybrid --mag_scale pwl \
  --sample_rate 24000 --fft_length 512 --spec_width 256 \
  --chunk_duration 3 --num_mels 64 --alpha 0.5 \
  --epochs 50 --checkpoint_path checkpoints/birdnet.keras
```

Optional accuracy boost (after the run above):

```bash
python -m birdnet_stm32 train --data_path_train data/train --qat \
  --checkpoint_path checkpoints/birdnet.keras --epochs 10 --learning_rate 0.0001
```

## 4. Convert to INT8 TFLite

```bash
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/birdnet.keras \
  --model_config checkpoints/birdnet_model_config.json \
  --data_path_train data/train
```

This writes `checkpoints/birdnet_quantized.tflite`. Confirm the reported cosine
similarity is > 0.95.

## 5. Embed into the firmware

```bash
# from this repo's root
python tools/convert_model.py \
  /path/to/birdnet-stm32/checkpoints/birdnet_quantized.tflite \
  main/model/model_data.cc
```

Then update [../main/model/labels.c](../main/model/labels.c) so the common/
scientific names are in the **same order** as the emitted `birdnet_labels.txt`.

## 6. Enable and build

Set `BIRDNET_HAS_REAL_MODEL` to `1` in
[../main/birdnet_config.h](../main/birdnet_config.h), then build + flash.

Watch the serial monitor at boot:

```
birdnet_model: model set: NNNN bytes, arena used=NNN KB, in=[1,257,256,1] type=... out=<num_classes>
```

- If it reports **arena too small**, raise `BIRDNET_TFLITE_ARENA_BYTES`.
- If it reports a **missing op**, add the matching `Add*()` in
  [../main/model/birdnet_model.cc](../main/model/birdnet_model.cc).

---

## Deploying multiple models on SD (model swap)

Instead of embedding one model in flash, you can put several on a microSD card
and cycle between them with a button (see the main README "Multiple models on
SD"). For each trained model:

1. Convert to INT8 TFLite as above (`birdnet_quantized.tflite`).
2. Copy it to `/sdcard/models/<name>.tflite` (FAT32).
3. Create `/sdcard/models/<name>.txt` from the emitted `*_labels.txt`. The
   firmware accepts one entry per line as either:
   - `Common Name|Scientific name`  (pipe or tab separated), or
   - a single name (used for both common and scientific).
   The order must match the model's class indices exactly.
4. List every model in `/sdcard/models/manifest.txt`
   (see [manifest.example.txt](manifest.example.txt)).

No `tools/convert_model.py` or rebuild is needed for SD models — only embedded
models use that path. Every SD model must share the same hybrid contract above.

---

## Verifying the frontend matches (optional but recommended)

A subtle risk is the on-device STFT differing from librosa's. To sanity-check,
compare one chunk:

1. In Python, take a 3 s/24 kHz mono WAV and run birdnet-stm32's
   `get_spectrogram_from_audio(..., mel_bins=0, mode="linear")` to get the
   `[257, 256]` magnitude it would feed the model.
2. Feed the same samples through the firmware and dump the `spectro` buffer
   (e.g. temporary `ESP_LOG` of a few bins) and compare.

Because the model self-normalizes (`y / max(y)` after a NonNeg mel mixer), small
scale/edge differences are largely absorbed — but matching the Hann window, FFT
size, and hop is what matters most, and those are fixed by the contract above.
