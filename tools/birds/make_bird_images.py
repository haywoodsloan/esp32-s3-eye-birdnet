#!/usr/bin/env python3
"""Fetch one representative photo per species and convert to the firmware's
RGB565 "BN16" image format for the ESP32-S3-EYE BirdNET detection overlay.

For each "Common|Scientific" line in the labels file this:
  1. Queries the Wikipedia "pageimages" API (by scientific name, then common
     name) for the article's lead image -- these are Wikimedia Commons photos,
     overwhelmingly public-domain / Creative-Commons licensed.
  2. Center-crops the photo to a square and resizes it to --size px.
  3. Encodes it as a tiny "BN16" file: b"BN16" + uint16 LE width + uint16 LE
     height + width*height RGB565 pixels (uint16 LE, R5 G6 B5 -- the exact bit
     order of the firmware's rgb565()).
  4. Writes <out>/<sanitized scientific name>.bin, where the name is lowercased
     with every non [a-z0-9] byte mapped to '_' (identical to the firmware's
     bird_filename() so the lookup matches).

A run also writes ATTRIBUTION.txt (species -> source image + article URL) next
to the images, and prints a summary of any species it could not fetch.

Usage (from the repo root):
    python tools/birds/make_bird_images.py            # fill in missing images
    python tools/birds/make_bird_images.py --force    # re-fetch everything
    python tools/birds/make_bird_images.py --only "Cardinalis cardinalis"

Requires: Pillow  (pip install Pillow)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_LABELS = os.path.join(REPO_ROOT, "sdcard", "models", "plainfield74.txt")
DEFAULT_OUT = os.path.join(REPO_ROOT, "sdcard", "birds")

# Wikimedia asks for a descriptive User-Agent identifying the tool.
USER_AGENT = ("esp32-s3-eye-birdnet/1.0 (detection-overlay asset builder; "
              "https://github.com/birdnet-team/birdnet-stm32)")
API = "https://en.wikipedia.org/w/api.php"


def sanitize(sci: str) -> str:
    """Lowercase; map every non [a-z0-9] byte to '_'. Matches the firmware."""
    out = []
    for ch in sci.strip():
        c = ch.lower()
        out.append(c if ("a" <= c <= "z" or "0" <= c <= "9") else "_")
    return "".join(out)


def http_get(url: str, params: dict | None = None, *, binary: bool = False,
             retries: int = 2, timeout: int = 30):
    """GET a URL (optionally with query params); return bytes or decoded JSON."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            return data if binary else json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report and retry/skip
            last = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise last  # type: ignore[misc]


def lead_image_url(title: str, thumb_px: int) -> str | None:
    """Return the lead-image (pageimages) thumbnail URL for an article, or None."""
    data = http_get(API, {
        "action": "query",
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": str(thumb_px),
        "titles": title,
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    })
    for page in data.get("query", {}).get("pages", []):
        thumb = page.get("thumbnail", {})
        if thumb.get("source"):
            return thumb["source"]
    return None


def article_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def to_square_rgb565(img: Image.Image, size: int) -> bytes:
    """Center-crop to square, resize to size x size, encode as BN16."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)

    rgb = img.tobytes()  # r,g,b,r,g,b,...
    out = bytearray(size * size * 2)
    j = 0
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[j] = v & 0xFF           # little-endian uint16
        out[j + 1] = (v >> 8) & 0xFF
        j += 2
    return b"BN16" + struct.pack("<HH", size, size) + bytes(out)


def load_labels(path: str) -> list[tuple[str, str]]:
    species = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                common, sci = line.split("|", 1)
            elif "\t" in line:
                common, sci = line.split("\t", 1)
            else:
                common = sci = line
            species.append((common.strip(), sci.strip()))
    return species


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="labels file (Common|Scientific)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory for .bin images")
    ap.add_argument("--size", type=int, default=232, help="square edge length in px "
                    "(232 = the overlay's full-bleed area; must be <= firmware "
                    "BIRDNET_BIRD_IMG_MAX_PX)")
    ap.add_argument("--force", action="store_true", help="re-fetch images that already exist")
    ap.add_argument("--only", default=None, help="only process this scientific or common name")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between API calls")
    args = ap.parse_args()

    species = load_labels(args.labels)
    if args.only:
        needle = args.only.lower()
        species = [s for s in species if needle in s[0].lower() or needle in s[1].lower()]
    if not species:
        print("No species to process.", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    thumb_px = max(320, args.size * 3)  # fetch larger, then downscale for quality

    done, skipped, failed = [], [], []
    attribution = []

    for idx, (common, sci) in enumerate(species, 1):
        fname = sanitize(sci) + ".bin"
        fpath = os.path.join(args.out, fname)
        tag = f"[{idx}/{len(species)}] {common} ({sci})"

        if os.path.exists(fpath) and not args.force:
            print(f"{tag}: skip (exists)")
            skipped.append(sci)
            continue

        try:
            # Prefer the scientific-name article; fall back to the common name.
            url = lead_image_url(sci, thumb_px)
            used_title = sci
            if not url:
                url = lead_image_url(common, thumb_px)
                used_title = common
            if not url:
                print(f"{tag}: FAILED (no lead image)")
                failed.append(sci)
                time.sleep(args.delay)
                continue

            raw = http_get(url, binary=True)
            blob = to_square_rgb565(Image.open(io.BytesIO(raw)), args.size)
            with open(fpath, "wb") as f:
                f.write(blob)
            print(f"{tag}: {fname} ({len(blob)} B) <- {url}")
            done.append(sci)
            attribution.append(f"{common}\t{sci}\t{article_url(used_title)}\t{url}")
        except Exception as exc:  # noqa: BLE001 - keep going on per-species errors
            print(f"{tag}: FAILED ({exc})")
            failed.append(sci)

        time.sleep(args.delay)

    if attribution:
        with open(os.path.join(args.out, "ATTRIBUTION.txt"), "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("# Common\tScientific\tArticle\tImage source (Wikipedia / Wikimedia Commons)\n")
            f.write("\n".join(attribution) + "\n")

    print("\n=== Summary ===")
    print(f"  written: {len(done)}   skipped: {len(skipped)}   failed: {len(failed)}")
    if failed:
        print("  failed species (re-run, or add a manual image):")
        for sci in failed:
            print(f"    - {sci}  -> {sanitize(sci)}.bin")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
