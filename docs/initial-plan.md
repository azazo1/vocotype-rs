# VocoType Rust 实现说明

## Summary
- 当前实现是单一 Rust 应用, 二进制名为 `vocotype`, 保留本地离线识别, 全局按键录音, 文本注入, 状态悬浮窗, 日志和数据集保存.
- 云端识别路径不进入 Rust 版本, 不保留云端凭据, WebSocket 识别或 Volcengine 分支.
- 模型不在 daemon 启动时自动下载. 缺模型时直接报错, 并提示运行 `vocotype models download --model-dir ...`.
- 按住转录键时使用 sherpa Silero VAD 做端点检测. 用户停顿达到阈值后立即提交当前语音段, 继续按住时继续录制下一段.

## Runtime
- `clap` 提供 `daemon`, `transcribe`, `models download`, `models doctor` 和 `devices` 子命令.
- `--model-dir`, `--model-cache-dir` 和 `--model-revision` 是 global args, 可以放在根命令后, 也可以放在子命令后.
- `cpal` 负责跨平台麦克风采集, 音频统一转成 16 kHz mono i16 PCM.
- `sherpa-onnx` 负责本地 ASR 和 VAD 推理, 发布包不需要用户安装 Python, FunASR, modelscope 或 ONNX Runtime.
- `global-hotkey` 负责全局快捷键, Linux 桌面完整体验以 X11 为目标.
- `eframe/egui` 做置顶状态悬浮窗, 显示 idle, recording, silence, transcribing, done, error 和 queue.
- `enigo` 优先做文本注入, 失败时用剪贴板回退, 并尽量恢复原剪贴板.
- `tokio`, `tracing`, `tracing-subscriber` 负责异步任务, 队列, 日志和阶段耗时.

## Models
- `--model-dir` 或 `VOCOTYPE_MODEL_DIR` 指定模型加载根目录.
- `--model-cache-dir` 或 `VOCOTYPE_MODEL_CACHE_DIR` 指定下载缓存目录.
- `--model-revision` 或 `VOCOTYPE_MODEL_REVISION` 目前默认记录为 `asr-models`, 对应 sherpa-onnx release tag.
- 下载后写入 `manifest.json`, 记录模型来源, 文件校验和和下载时间.
- ASR 使用 `sherpa-onnx-paraformer-zh-2024-03-09`.
- ASR 加载时优先使用 `model.onnx`, 其次使用 `model.int8.onnx`, 也会扫描目录下的 `.onnx` 文件.
- VAD 使用 `silero_vad.onnx`.
- VAD 下载后校验 sha256, 防止缓存文件损坏.
- `models doctor` 会输出模型目录, 缓存目录, manifest 状态和缺失提示. 模型齐全时还会实际加载 sherpa ASR 和 VAD.
- PUNC 没有作为独立必需模型保留, 避免留下未接入运行时的冗余模型路径.

## VAD
- daemon 进入按住录音状态后持续接收音频块.
- 每个音频块送入 sherpa `VoiceActivityDetector`.
- VAD 输出语音段后立即推入转写队列, 不等待热键释放.
- `pre_roll_ms` 和 `tail_padding_ms` 用于扩展 VAD 输出边界.
- `min_speech_ms`, `end_silence_ms` 和 `max_segment_ms` 传给 Silero VAD 配置.
- 松开热键时调用 VAD flush, 只提交剩余有效语音段.

## Manual Checks
- 已用缺模型目录验证 `models doctor --model-dir ...` 会输出下载提示并失败退出.
- 已通过 `cargo build`, `cargo clippy` 和 `cargo test`.
- 仍需在 macOS, Windows, Linux X11 分别验证麦克风权限, 全局热键, 悬浮窗置顶, 文本注入和剪贴板回退.
- 仍需下载真实模型后, 用中文短音频验证 `transcribe --audio` 和按住热键分段注入效果.
