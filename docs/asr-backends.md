# ASR 后端对比

VocoType 提供 `sherpa-onnx` 和讯飞 `EdgeEsr` 两个本地 ASR 后端. 本文比较仓库当前集成的默认模型和实现, 不代表底层框架全部可选模型的能力.

## 能力对比

| 能力 | sherpa-onnx | 讯飞 EdgeEsr |
| --- | --- | --- |
| 默认后端 | 是 | 否 |
| WAV 文件转写 | 支持 | 支持 |
| SRT 字幕输出 | 支持 | 支持 |
| 麦克风录音 | 支持 | 支持 |
| VAD | Silero VAD | 讯飞 `vad.onnx` |
| 实时流式输入 | 不支持 | 支持 |
| GUI partial 结果 | 不支持 | 支持稳定文本和待修正文本 |
| 流式结果修正 | 不支持 | 支持 `revision` |
| 输入框注入 | `final` 一次性注入 | `final` 一次性注入 |
| 标点恢复 | 独立 sherpa PUNC 模型 | 讯飞后处理链 |
| 词典文本改写 | 支持 | 支持 |
| `hotwords_score` 参数 | 已传入 sherpa, 但默认 Paraformer 不使用 | 不支持, 参数无效 |
| 解码热词加权 | 当前默认模型不支持 | 当前未实现 |
| 模型检查和下载 | 支持 | 支持 |
| 模型组成 | ASR + VAD + PUNC | EdgeEsr 模型和后处理数据 |
| custom-op | 不涉及 | Rust 实现并编译进可执行文件 |

## 选择后端

- `sherpa-onnx` 是默认后端, 适合只需要离线分段转写和独立标点模型的场景.
- 讯飞 `EdgeEsr` 适合需要在录音期间看到实时结果和修正结果的场景.

两个后端都可用于 WAV 转写, SRT 输出, 热键录音和最终文本注入. 后端差异主要在实时流式输出, 模型包构成和解码热词能力.

## 词汇表与解码热词

`hotwords` 是最终文本的大小写和拆字母归一化. 例如, 它可以把 `gpt`, `g p t` 和 `open ai` 归一化为 `GPT` 和 `OpenAI`.

`rewrites` 是确定性的文本替换. 它根据词汇表中的显式规则替换识别文本, 适合处理已知的误识别或逐字母结果.

`hotwords_score` 是解码阶段对候选路径的加权分数. VocoType 会把该参数传给 sherpa-onnx, 但当前默认 Paraformer 模型不支持 contextual biasing, 因而不会使用该分数. 讯飞 EdgeEsr 当前没有对应实现, 该参数对它无效.

因此, 当前两个默认模型都不能利用 `hotwords_score` 在解码时提高特定词的候选权重. 需要稳定修正最终文本时, 应使用 `hotwords` 或 `rewrites`.

## 流式录音

只有讯飞 EdgeEsr 后端会在麦克风录音期间接收实时流式输入. 它持续产生 partial 结果, 当新结果改写已有文本时标记为 `revision`.

GUI 会用这些结果实时显示稳定文本和待修正文本. 输入框不会注入 partial 或 revision, 只会在收到 `final` 结果时一次性注入最终文本.

`sherpa-onnx` 仍会在录音时通过 VAD 切分语音段, 但每个语音段都按离线方式解码, 只产生一次 final 结果, 不产生 partial 或 revision.

实时行为只适用于热键录音. WAV 文件转写最终仍输出完整的转写或 SRT 结果.

## 模型检查与下载

默认命令操作 sherpa-onnx 模型:

```shell
vocotype models download
vocotype models doctor
```

讯飞 EdgeEsr 使用独立的模型包:

```shell
vocotype models download --backend iflytek
vocotype models doctor --backend iflytek
```

Sherpa 模型由 ASR, Silero VAD 和独立 PUNC 模型组成. 讯飞模型包包含 EdgeEsr 的 ASR ONNX 模型, `vad.onnx`, 标点恢复和数字归一化等后处理数据. 所需 custom-op 由 Rust 实现并随可执行文件编译.
