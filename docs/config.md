# 配置文件

VocoType 默认读取 `~/.config/vocotype/config.toml`.

可以用下面的命令查看默认模板:

```shell
vocotype config default
```

可以用下面的命令查看 JSON Schema:

```shell
vocotype config schema
```

## 用户词汇表

VocoType 默认读取 `~/.config/vocotype/dict.toml` 作为用户词汇表. `config.toml` 只需要在自定义位置时配置 `dict-path`.

```toml
dict-path = "~/.config/vocotype/dict.toml"

[asr]
hotwords-score = 3.0
```

`hotwords-score` 只对支持 sherpa contextual biasing 的模型有效. 当前默认 Paraformer 模型会使用词表后处理, 不调用 sherpa hotwords stream.

可以用下面的命令输出 `dict.toml` 模板:

```shell
vocotype dict default
```

可以用下面的命令检查词汇表是否加载:

```shell
vocotype dict doctor
```

`dict.toml` 支持 `hotwords`, `rewrites` 和 `rime-imports`. `hotwords` 用于后处理归一化英文热词, `rewrites` 会在标点恢复后做显式文本改写. 例如逐个字母读出 `g p t` 时, 可以改写为 `GPT`.

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

默认只导入 ASCII 英文词条.

## 热键

`daemon.hotkey` 支持单键和组合键.

单键示例:

```toml
[daemon]
hotkey = "F2"
```

组合键使用 `+` 连接, 修饰键写在前面, 主键写在最后:

```toml
[daemon]
hotkey = "ctrl+f2"
```

也可以写多个修饰键:

```toml
[daemon]
hotkey = "shift+alt+space"
```

常用修饰键包括 `ctrl`, `control`, `shift`, `alt`, `option`, `cmd`, `command`, `super` 和 `cmdorctrl`.

主键可以写功能键, 字母键和常见控制键, 例如 `F2`, `KeyQ`, `space`, `enter`, `escape`, `left`.

## 热键模式

`daemon.hotkey-mode` 控制录音开始和停止的方式.

按住录音, 松开停止:

```toml
[daemon]
hotkey = "ctrl+f2"
hotkey-mode = "pressed"
```

按一次开始, 再按一次停止:

```toml
[daemon]
hotkey = "ctrl+f2"
hotkey-mode = "toggle"
```

使用触发键开始, 使用结束键停止:

```toml
[daemon]
hotkey = "ctrl+f2"
hotkey-mode = "trigger-end"
end-hotkey = "ctrl+f3"
```

`trigger-end` 模式必须配置 `end-hotkey`. `end-hotkey` 的写法和 `hotkey` 相同, 并且不能和 `hotkey` 相同.

## 示例

```toml
dict-path = "~/.config/vocotype/dict.toml"
model-revision = "asr-models"

[asr]
hotwords-score = 3.0

[daemon]
hotkey = "ctrl+f2"
hotkey-mode = "pressed"
# end-hotkey = "ctrl+f3"
save-dataset = false
append-newline = false
strip-trailing-period = false
inject-method = "auto"
end-silence-ms = 650
pre-roll-ms = 180
tail-padding-ms = 180
min-speech-ms = 240
max-segment-ms = 15000
idle-unload-secs = 300

[transcribe]
format = "json"
pretty = false
subtitle-max-chars = 24
```
