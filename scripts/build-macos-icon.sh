#!/bin/sh
set -eu

SVG_PATH="assets/app-icon.svg"
PNG_PATH="assets/app-icon.png"
ICNS_PATH="assets/app-icon.icns"
ICONSET_DIR="target/macos-icon/app-icon.iconset"

if ! command -v rsvg-convert >/dev/null 2>&1; then
    printf '%s\n' "rsvg-convert is required to build the app icon" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "uv is required to build the app icon" >&2
    exit 1
fi

mkdir -p "$ICONSET_DIR"

rsvg-convert -w 1024 -h 1024 "$SVG_PATH" -o "$PNG_PATH"

for size in 16 32 128 256 512; do
    rsvg-convert -w "$size" -h "$size" "$SVG_PATH" -o "$ICONSET_DIR/icon_${size}x${size}.png"
    scale_size=$((size * 2))
    rsvg-convert -w "$scale_size" -h "$scale_size" "$SVG_PATH" -o "$ICONSET_DIR/icon_${size}x${size}@2x.png"
done

uv run python - "$ICONSET_DIR" "$ICNS_PATH" <<'PY'
from pathlib import Path
import struct
import sys

iconset = Path(sys.argv[1])
out = Path(sys.argv[2])
entries = [
    ("icp4", "icon_16x16.png"),
    ("icp5", "icon_32x32.png"),
    ("icp6", "icon_32x32@2x.png"),
    ("ic07", "icon_128x128.png"),
    ("ic08", "icon_256x256.png"),
    ("ic09", "icon_512x512.png"),
    ("ic10", "icon_512x512@2x.png"),
    ("ic11", "icon_16x16@2x.png"),
    ("ic12", "icon_32x32@2x.png"),
    ("ic13", "icon_128x128@2x.png"),
    ("ic14", "icon_256x256@2x.png"),
]
chunks = []
for code, name in entries:
    data = (iconset / name).read_bytes()
    chunks.append(code.encode("ascii") + struct.pack(">I", len(data) + 8) + data)
blob = b"icns" + struct.pack(">I", 8 + sum(len(chunk) for chunk in chunks)) + b"".join(chunks)
out.write_bytes(blob)
PY
printf '%s\n' "$ICNS_PATH"
