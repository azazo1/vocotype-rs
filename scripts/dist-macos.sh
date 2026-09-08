#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

APP_NAME="VocoType"
BIN_NAME="vocotype"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
APP_ROOT="$TARGET_DIR/macos-app"
APP_DIR="$APP_ROOT/$APP_NAME.app"
DMG_STAGE="$APP_ROOT/dmg-stage"
DMG_PATH="$APP_ROOT/$APP_NAME.dmg"
APP_VERSION="${APP_VERSION:-$(cargo metadata --locked --no-deps --format-version 1 | jq -er '.packages[] | select(.name == "vocotype-rs") | .version')}"
BUILD_VERSION="${BUILD_VERSION:-$APP_VERSION}"

uname_m="$(uname -m)"
case "$uname_m" in
    arm64)
        arch="aarch64"
        ;;
    x86_64)
        arch="x86_64"
        ;;
    *)
        printf '不支持的 macOS 架构: %s\n' "$uname_m" >&2
        exit 1
        ;;
esac

sh scripts/package-macos-app.sh

if [ ! -d "$APP_DIR" ]; then
    printf '缺少 app bundle: %s\n' "$APP_DIR" >&2
    exit 1
fi
if [ ! -x "$APP_DIR/Contents/MacOS/$BIN_NAME" ]; then
    printf '缺少 app 可执行文件: %s\n' "$APP_DIR/Contents/MacOS/$BIN_NAME" >&2
    exit 1
fi

printf '正在创建 %s 磁盘镜像...\n' "$APP_NAME"
rm -rf "$DMG_STAGE" "$DMG_PATH"
mkdir -p "$DMG_STAGE"
ditto "$APP_DIR" "$DMG_STAGE/$APP_NAME.app"
ln -s /Applications "$DMG_STAGE/Applications"
hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGE" \
    -ov \
    -format UDZO \
    -fs HFS+ \
    "$DMG_PATH"
rm -rf "$DMG_STAGE"

if [ ! -s "$DMG_PATH" ]; then
    printf '创建磁盘镜像失败: %s\n' "$DMG_PATH" >&2
    exit 1
fi

mkdir -p dist
dist_dmg="dist/vocotype-${BUILD_VERSION}-macos-${arch}.dmg"
cp "$DMG_PATH" "$dist_dmg"
printf '%s\n' "$dist_dmg"
