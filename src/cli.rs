use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{Shell, generate};

use crate::asr::AsrEngine;
use crate::audio::list_input_devices;
use crate::daemon::{DaemonOptions, run_daemon};
use crate::inject::InjectMethod;
use crate::models::{DEFAULT_REVISION, ModelOptions, ModelStore};
use crate::subtitle::{SubtitleOptions, transcribe_srt};

#[derive(Parser, Debug)]
#[command(
    name = "vocotype",
    version,
    about = "本地语音转写和文本注入工具",
    long_about = "VocoType 使用本地 sherpa-onnx 模型完成录音转写, VAD 分段, 标点恢复, 文本注入和 SRT 字幕输出."
)]
pub struct Cli {
    #[arg(
        long,
        env = "VOCOTYPE_MODEL_DIR",
        global = true,
        help = "指定模型加载目录",
        long_help = "指定 ASR, VAD 和 PUNC 模型的加载根目录. 也可以通过 VOCOTYPE_MODEL_DIR 设置."
    )]
    pub model_dir: Option<PathBuf>,

    #[arg(
        long,
        env = "VOCOTYPE_MODEL_CACHE_DIR",
        global = true,
        help = "指定模型下载缓存目录",
        long_help = "指定 models download 写入下载缓存和模型文件的目录. 也可以通过 VOCOTYPE_MODEL_CACHE_DIR 设置."
    )]
    pub model_cache_dir: Option<PathBuf>,

    #[arg(
        long,
        env = "VOCOTYPE_MODEL_REVISION",
        default_value = DEFAULT_REVISION,
        global = true,
        help = "指定模型版本标签",
        long_help = "指定模型 manifest 记录的 revision. 默认使用当前 sherpa-onnx release 标签."
    )]
    pub model_revision: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    #[command(alias = "d", about = "启动热键监听和悬浮窗")]
    Daemon(DaemonArgs),
    #[command(alias = "t", about = "转写音频文件或生成 SRT 字幕")]
    Transcribe(TranscribeArgs),
    #[command(alias = "m", about = "下载和检查本地模型")]
    Models(ModelsCommand),
    #[command(about = "列出可用麦克风输入设备")]
    Devices,
    #[command(alias = "comp", about = "生成 shell 补全脚本")]
    Completion(CompletionArgs),
}

#[derive(Args, Debug)]
pub struct DaemonArgs {
    #[arg(
        long,
        default_value = "F2",
        env = "VOCOTYPE_HOTKEY",
        help = "按住录音的全局热键"
    )]
    pub hotkey: String,

    #[arg(long, default_value_t = false, help = "保存转写音频和结果用于数据集")]
    pub save_dataset: bool,

    #[arg(
        long,
        env = "VOCOTYPE_DATASET_DIR",
        help = "指定数据集保存目录"
    )]
    pub dataset_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false, help = "注入文本后追加换行")]
    pub append_newline: bool,

    #[arg(
        long,
        env = "VOCOTYPE_INJECT_METHOD",
        default_value = "auto",
        help = "选择文本注入方式",
        long_help = "选择文本注入方式. 可用值由 inject 模块解析, 常用值包括 auto, enigo 和 clipboard."
    )]
    pub inject_method: String,

    #[arg(long, default_value_t = 650, help = "判定语音结束的静音毫秒数")]
    pub end_silence_ms: u32,

    #[arg(long, default_value_t = 180, help = "分段前保留的音频毫秒数")]
    pub pre_roll_ms: u32,

    #[arg(long, default_value_t = 180, help = "分段后追加的音频毫秒数")]
    pub tail_padding_ms: u32,

    #[arg(long, default_value_t = 240, help = "保留语音段的最短毫秒数")]
    pub min_speech_ms: u32,

    #[arg(long, default_value_t = 15_000, help = "单个语音段的最长毫秒数")]
    pub max_segment_ms: u32,

    #[arg(
        long,
        default_value_t = 300,
        help = "空闲多少秒后卸载 ASR 和 PUNC 模型",
        long_help = "daemon 队列空闲达到该秒数后卸载 ASR 和 PUNC 模型以降低内存占用. 设置为 0 表示不自动卸载."
    )]
    pub idle_unload_secs: u64,
}

#[derive(Args, Debug)]
pub struct TranscribeArgs {
    #[arg(index = 1, help = "要转写的 WAV 音频文件")]
    pub audio: PathBuf,

    #[arg(
        short,
        long,
        value_enum,
        default_value_t = TranscribeFormat::Json,
        help = "选择输出格式"
    )]
    pub format: TranscribeFormat,

    #[arg(long, default_value_t = false, conflicts_with = "json", help = "等效于 --format srt")]
    pub srt: bool,

    #[arg(long, default_value_t = false, help = "等效于 --format json")]
    pub json: bool,

    #[arg(short, long, help = "把转写结果写入文件")]
    pub output: Option<PathBuf>,

    #[arg(long, default_value_t = false, help = "以 pretty JSON 输出转写结果")]
    pub pretty: bool,

    #[arg(long, default_value_t = 24, help = "SRT 每条字幕的目标最大字符数")]
    pub subtitle_max_chars: usize,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum TranscribeFormat {
    #[value(help = "输出 JSON 转写结果")]
    Json,
    #[value(help = "输出 SRT 字幕")]
    Srt,
}

impl TranscribeArgs {
    fn resolved_format(&self) -> TranscribeFormat {
        if self.srt {
            TranscribeFormat::Srt
        } else if self.json {
            TranscribeFormat::Json
        } else {
            self.format.clone()
        }
    }
}

#[derive(Args, Debug)]
pub struct ModelsCommand {
    #[command(subcommand)]
    pub command: ModelsSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelsSubcommand {
    #[command(about = "下载 ASR, VAD 和 PUNC 模型")]
    Download,
    #[command(about = "检查模型目录和运行时加载能力")]
    Doctor,
}

#[derive(Args, Debug)]
pub struct CompletionArgs {
    #[arg(value_enum, help = "要生成补全脚本的 shell")]
    pub shell: Shell,
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    let options = ModelOptions {
        model_dir: cli.model_dir.clone(),
        model_cache_dir: cli.model_cache_dir.clone(),
        revision: cli.model_revision.clone(),
    };
    let store = ModelStore::new(&options);

    match cli.command {
        Command::Daemon(args) => {
            let daemon = DaemonOptions {
                hotkey: args.hotkey,
                save_dataset: args.save_dataset,
                dataset_dir: args.dataset_dir,
                append_newline: args.append_newline,
                inject_method: InjectMethod::parse(&args.inject_method),
                end_silence_ms: args.end_silence_ms,
                pre_roll_ms: args.pre_roll_ms,
                tail_padding_ms: args.tail_padding_ms,
                min_speech_ms: args.min_speech_ms,
                max_segment_ms: args.max_segment_ms,
                idle_unload_secs: args.idle_unload_secs,
            };
            run_daemon(store, daemon).await
        }
        Command::Transcribe(args) => {
            match args.resolved_format() {
                TranscribeFormat::Json => {
                    let engine = AsrEngine::load(store)?;
                    let result = engine.transcribe_file(&args.audio)?;
                    if args.pretty {
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!("{}", serde_json::to_string(&result)?);
                    }
                }
                TranscribeFormat::Srt => {
                    let srt = transcribe_srt(
                        store,
                        &args.audio,
                        SubtitleOptions {
                            max_chars: args.subtitle_max_chars,
                        },
                    )?;
                    if let Some(output) = args.output {
                        crate::app::ensure_parent(&output)?;
                        std::fs::write(&output, srt)?;
                    } else {
                        print!("{}", srt);
                    }
                }
            }
            Ok(())
        }
        Command::Models(models) => match models.command {
            ModelsSubcommand::Download => {
                let manifest = store.download_all().await?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
                Ok(())
            }
            ModelsSubcommand::Doctor => {
                crate::models::write_doctor_report(&store, std::io::stdout())?;
                crate::models::loadability_report(&store, std::io::stdout())?;
                Ok(())
            }
        },
        Command::Devices => {
            for device in list_input_devices()? {
                println!("{}", device);
            }
            Ok(())
        }
        Command::Completion(args) => {
            let mut command = Cli::command();
            generate(args.shell, &mut command, "vocotype", &mut std::io::stdout());
            Ok(())
        }
    }
}
