# VocoType

VocoType 是一个本地语音转写和文本注入工具. 默认使用 sherpa-onnx, 也可以选择纯 Rust 接入的讯飞 EdgeEsr 后端完成语音识别, VAD 分段, 标点恢复, 热键录音, 文本注入和 SRT 字幕输出.

本项目是 [vocotype-cli](https://github.com/233stone/vocotype-cli) 的 Rust 实现版本.

## 特性

- 本地运行, 语音数据不需要发送到云端.
- 全局热键录音, 支持单键和 `ctrl+f2` 这类组合键.
- 支持 `pressed`, `toggle`, `trigger-end` 三种热键模式.
- 录音后自动转写并注入到当前输入位置.
- 可配置注入时追加换行或删除末尾句号.
- 支持 WAV 文件转写和 SRT 字幕生成.
- 支持 TOML 配置文件, JSON Schema 和配置生效检查.
- 支持用户词汇表, hotwords 和逐字母短语改写.
- 可从 Rime 英文 .dict.yaml 导入热词.
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

下载并检查讯飞模型:

```shell
vocotype models download --backend iflytek
vocotype models doctor --backend iflytek
```

讯飞模型不提交到 Git 或 Git LFS, 由当前仓库的独立 `models-iflytek-v1.0.0` Release 提供. 模型包只包含 ONNX 模型和后处理数据, Rust custom-op 直接编译进 `vocotype` 可执行文件.

使用讯飞后端转写:

```shell
vocotype --asr-backend iflytek transcribe input.wav
```

生成可编辑的词汇表模板:

```shell
vocotype dict default
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

## 用户词汇表

默认词汇表路径是 `~/.config/vocotype/dict.toml`. 可以用 `vocotype dict default` 输出模板, 再保存为自己的 `dict.toml`.

词表支持三类内容:

- `hotwords`: 用于后处理归一化英文热词, 例如把 `gpt`, `g p t`, `open ai` 改成 `GPT`, `OpenAI`.
- `rewrites`: 把逐字母转写结果改成目标写法, 例如 `g p t` 到 `GPT`.
- `rime-imports`: 从 Rime 英文 `.dict.yaml` 导入 hotwords.

当前默认 Paraformer ASR 模型不支持 sherpa contextual biasing, 所以 hotwords 不会直接改变 decoder 搜索结果. 如果需要处理更自由的误识别, 用 `[rewrites]` 写显式替换规则.

示例:

```toml
hotwords = [
  "GPT",
  "ChatGPT",
  "OpenAI",
  "API",
]

rime-imports = ["~/Library/Rime/wanxiang_english.dict.yaml"]
max-rime-words = 20000

[rewrites]
"g p t" = "GPT"
"a p i" = "API"
"u r l" = "URL"
```

Rime `.dict.yaml` 路径示例:

- macOS: `~/Library/Rime/wanxiang_english.dict.yaml`
- Linux fcitx5: `~/.local/share/fcitx5/rime/wanxiang_english.dict.yaml`
- Linux ibus: `~/.config/ibus/rime/wanxiang_english.dict.yaml`
- Windows: `C:/Users/Alice/AppData/Roaming/Rime/wanxiang_english.dict.yaml`

默认只导入英文词条.

可参考的 Rime 词库来源:

- [rime-wanxiang dicts](https://github.com/amzxyz/rime-wanxiang/tree/wanxiang/dicts)
- [oh-my-rime dicts](https://github.com/Mintimate/oh-my-rime/tree/main/dicts)

下载后把需要的 `.dict.yaml` 路径加入 `rime-imports`. VocoType 会读取 `import_tables` 引用的词库文件.

检查词汇表是否加载:

```shell
vocotype dict doctor
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

启用后处理英文标点模式:

```toml
[post-processing]
english-punctuation = true
strip-trailing-period = true
```

选择讯飞后端:

```toml
[asr]
backend = "iflytek"
```

这些选项会在词表改写之后处理最终转写文本, 可把中文标点转换为 ASCII 标点, 也可删除末尾句号.

更多配置示例见 [docs/config.md](docs/config.md).

## 常用命令

```shell
vocotype devices
vocotype models download
vocotype models doctor
vocotype models doctor --backend iflytek
vocotype daemon
vocotype transcribe input.wav --format srt
vocotype dict doctor
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
