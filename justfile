default:
    @just --list

# 构建项目.
build:
    cargo build

# 运行 clippy.
clippy:
    cargo clippy

# 运行测试.
test:
    cargo test

# 生成 macOS app icon 资产.
build-icon:
    sh scripts/build-macos-icon.sh

# just run-daemon --hotkey F2
# 启动 daemon.
run-daemon *args:
    cargo run -- daemon {{args}}

# 生成 macOS .app bundle, 输出到 target/macos-app/VocoType.app.
macos-app: build-icon
    sh scripts/package-macos-app.sh

# just download-models --model-dir ./models
# 下载模型.
download-models *args:
    cargo run -- models download {{args}}
