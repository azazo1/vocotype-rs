# VocoType

VocoType 是一个本地语音转写和文本注入工具. 它使用本地 sherpa-onnx 模型完成语音识别, VAD 分段, 标点恢复, 热键录音, 文本注入和 SRT 字幕输出.

本项目是 [vocotype-cli](https://github.com/233stone/vocotype-cli) 的 Rust 实现版本.

## 特性

- 本地运行, 语音数据不需要发送到云端.
- 全局热键录音, 支持单键和 `ctrl+f2` 这类组合键.
- 支持 `pressed`, `toggle`, `trigger-end` 三种热键模式.
- 录音后自动转写并注入到当前输入位置.
- 支持 WAV 文件转写和 SRT 字幕生成.
- 支持 TOML 配置文件, JSON Schema 和配置生效检查.
- 空闲时可自动卸载 ASR 和 PUNC 模型, 降低内存占用.

## 快速开始

安装或构建后, 先下载模型:

```shell
vocotype models download
```

检查模型是否可用:

```shell
vocotype models doctor
```

启动热键监听:

```shell
vocotype daemon --hotkey F2
```

默认模式是 `pressed`, 按住热键录音, 松开后停止并转写.

## 热键模式

按住录音:

```shell
vocotype daemon --hotkey "ctrl+f2" --hotkey-mode pressed
```

按一次开始, 再按一次停止:

```shell
vocotype daemon --hotkey "ctrl+f2" --hotkey-mode toggle
```

触发键开始, 结束键停止:

```shell
vocotype daemon --hotkey "ctrl+f2" --hotkey-mode trigger-end --end-hotkey "ctrl+f3"
```

## 转写文件

输出 JSON:

```shell
vocotype transcribe input.wav
```

输出 SRT:

```shell
vocotype transcribe input.wav --format srt --output input.srt
```

## 配置文件

默认配置路径是 `~/.config/vocotype/config.toml`.

输出默认模板:

```shell
vocotype config default
```

输出 JSON Schema:

```shell
vocotype config schema
```

检查配置是否加载并生效:

```shell
vocotype config doctor
```

更多配置示例见 [docs/config.md](docs/config.md).

## 常用命令

```shell
vocotype devices
vocotype models download
vocotype models doctor
vocotype daemon
vocotype transcribe input.wav --format srt
vocotype config doctor
vocotype completion zsh
```

## 开发

```shell
cargo build
cargo clippy
cargo test
```

也可以使用项目里的 `justfile`:

```shell
just build
just clippy
just test
just run-daemon --hotkey F2
```
