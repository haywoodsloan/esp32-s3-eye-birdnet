#!/usr/bin/env python3
"""Decode a firmware "BN16" RGB565 image back to a PNG, to verify the bytes
the ESP32 will read render as a correct, right-colored photo.

Usage:
    python tools/birds/preview_bin.py sdcard/birds/turdus_migratorius.bin [out.png]
"""
import struct
import sys

from PIL import Image


def decode(path: str) -> Image.Image:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"BN16":
        raise ValueError(f"{path}: not a BN16 file")
    w, h = struct.unpack("<HH", data[4:8])
    px = data[8:]
    if len(px) != w * h * 2:
        raise ValueError(f"{path}: expected {w*h*2} pixel bytes, got {len(px)}")
    img = Image.new("RGB", (w, h))
    out = img.load()
    i = 0
    for y in range(h):
        for x in range(w):
            v = px[i] | (px[i + 1] << 8)  # little-endian uint16
            i += 2
            r = ((v >> 11) & 0x1F) << 3
            g = ((v >> 5) & 0x3F) << 2
            b = (v & 0x1F) << 3
            out[x, y] = (r, g, b)
    return img


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
    img = decode(src)
    img.save(dst)
    print(f"{src} -> {dst} ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
