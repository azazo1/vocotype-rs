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

# just run-daemon --hotkey F2
# 启动 daemon.
run-daemon *args:
    cargo run -- daemon {{args}}

# just download-models --model-dir ./models
# 下载模型.
download-models *args:
    cargo run -- models download {{args}}
