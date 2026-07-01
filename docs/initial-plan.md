# VocoType Rust 移植计划

## Summary
- 将当前空壳 Rust 项目改造成单一 Rust 应用, 保留本地 FunASR 离线识别, 全局按键录音, 文本注入, 状态悬浮窗, 日志, 数据集保存等主要体验.
- 删除云端 Volcengine 分支, 不再保留任何云端凭据或 WebSocket 识别路径.
- 模型不在 daemon 启动时自动下载. 新增专门的 `models download` 子命令, daemon 缺模型时直接报错并提示用户运行下载命令.
- 按住转录键期间启用 VAD 分段. 用户停顿达到阈值后立即提交当前语音段转写, 继续按住时下一段继续录制, 松开时只提交剩余未提交片段.

## Key Changes
- CLI 使用 `clap` 设计子命令:
  - `vocotype daemon`: 启动热键, 音频采集, VAD 分段, 悬浮窗和文本注入.
  - `vocotype transcribe --audio <path>`: 对本地音频文件做一次转录.
  - `vocotype models download`: 下载 ASR, VAD, PUNC 三类 FunASR ONNX 模型.
  - `vocotype models doctor`: 检查模型目录, ONNX Runtime 可加载性, 音频设备和平台权限状态.
- 模型路径支持参数和环境变量:
  - `--model-dir` / `VOCOTYPE_MODEL_DIR` 指定模型加载根目录.
  - `--model-cache-dir` / `VOCOTYPE_MODEL_CACHE_DIR` 指定下载缓存目录, 未设置时默认使用用户缓存目录下的 `vocotype/models`.
  - 下载后写入 `manifest.json`, 记录模型名, revision, 文件校验和和下载时间.
- 模型默认值沿用原项目:
  - revision: `v2.0.5`.
  - ASR: `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`.
  - VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-onnx`.
  - PUNC: `iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx`.
- Rust 运行时结构:
  - `cpal` 负责跨平台麦克风采集, 统一转成 16 kHz mono i16 PCM.
  - `ort` 负责 ONNX Runtime 推理, 发布包随平台带上 ONNX Runtime 原生库, 用户不需要安装 Python, FunASR, modelscope 或命令行工具.
  - `global-hotkey` 负责全局快捷键, Linux 完整支持范围按 X11 处理.
  - `eframe/egui` 做置顶状态悬浮窗, 显示 idle, recording, silence, transcribing, done, error, queue.
  - `enigo` 优先做文本注入, 失败时用剪贴板回退, 并尽量恢复原剪贴板.
  - `tokio`, `tracing`, `tracing-subscriber` 负责异步任务, 队列, 日志和阶段耗时.
- VAD 分段行为:
  - daemon 进入按住录音状态后持续接收音频块.
  - 使用轻量本地端点检测先判定静音窗口, 达到 `end_silence_ms` 后封段并推入转写队列.
  - 每段保留 `pre_roll_ms` 和 `tail_padding_ms`, 避免切掉开头和结尾.
  - `min_speech_ms` 以下的段丢弃, `max_segment_ms` 到达后强制封段.
  - 松开热键时停止采集, 如果当前段有有效语音则提交, 如果只有静音则丢弃.
- 悬浮窗行为:
  - daemon 启动时创建无边框小窗, 置顶, 默认不抢焦点.
  - 录音时显示电平和当前状态, 静音等待时显示 pending, 转写时显示队列数量, 成功后短暂显示识别文本预览.
  - 错误时显示明确操作提示, 例如缺模型时提示 `vocotype models download --model-dir ...`.
- 文件组织:
  - `src/main.rs` 只保留入口和 CLI dispatch.
  - 新增 `audio`, `vad`, `asr`, `models`, `daemon`, `overlay`, `hotkey`, `inject`, `dataset`, `logging` 等模块.
  - 补充 `justfile`, 提供 `just build`, `just clippy`, `just test`, `just run-daemon`, `just download-models`.

## Test Plan
- 单元测试:
  - 模型路径解析和参数/环境变量优先级.
  - VAD 分段状态机, 覆盖静音丢弃, 停顿封段, 松开提交, 最大时长强制封段.
  - 下载 manifest 校验和缺文件报错提示.
- 集成测试:
  - 使用短 WAV fixture 跑 `transcribe --audio`, 校验返回结构和非空文本, 不做文案精确断言.
  - 使用模拟音频流验证 daemon 的分段队列行为.
  - `models doctor` 在缺模型目录时返回可读错误和下载命令提示.
- 手动验收:
  - macOS, Windows, Linux X11 分别验证麦克风权限, 全局热键, 悬浮窗置顶, 文本注入和剪贴板回退.
  - 按住热键说两句话, 中间停顿, 确认第一句会在未松开时先转写并注入.
  - 无模型启动 daemon, 确认不会下载, 只提示运行 `models download`.

## Assumptions
- "无需外部依赖" 按发布包自带原生推理库理解, 用户机器不需要安装 Python, FunASR, modelscope 或 ONNX Runtime.
- "vae" 按 VAD/端点检测理解.
- Linux 桌面完整体验以 X11 为目标. Wayland 下热键和文本注入按受限环境处理, `models` 和 `transcribe` 仍可用.
- FunASR Rust 侧会优先复刻原项目的 ONNX 推理效果. 若 FunASR Python 前后处理存在未公开细节, 实现时以模型真实输入输出和少量音频样本对齐为准.
- 主要依赖选择参考官方文档: [cpal](https://docs.rs/cpal/latest/cpal/), [ort](https://docs.rs/ort/latest/ort/), [global-hotkey](https://docs.rs/global-hotkey/latest/global_hotkey/), [enigo](https://docs.rs/enigo/latest/enigo/), [eframe](https://docs.rs/eframe/latest/eframe/struct.NativeOptions.html).
