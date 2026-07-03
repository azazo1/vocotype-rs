use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use clap::parser::ValueSource;
use clap::{ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use clap_complete::{Shell, generate};

use crate::asr::{AsrEngine, AsrOptions};
use crate::audio::list_input_devices;
use crate::config::{AppConfig, config_schema, default_config_path, default_config_template};
use crate::daemon::{DaemonOptions, HotkeyMode, run_daemon};
use crate::dict::{
    DEFAULT_HOTWORDS_SCORE, SpeechDictionary, default_dict_template, write_dict_doctor,
};
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

    #[arg(
        long,
        env = "VOCOTYPE_CONFIG",
        global = true,
        help = "指定配置文件路径",
        long_help = "指定 TOML 配置文件路径. 默认读取 ~/.config/vocotype/config.toml, 文件不存在时跳过."
    )]
    pub config: Option<PathBuf>,

    #[arg(
        long = "dict",
        env = "VOCOTYPE_DICT",
        global = true,
        help = "指定用户词汇表路径",
        long_help = "指定 dict.toml 词汇表路径. 默认读取 ~/.config/vocotype/dict.toml, 文件不存在时使用内置词汇表."
    )]
    pub dict: Option<PathBuf>,

    #[arg(
        long,
        env = "VOCOTYPE_HOTWORDS_SCORE",
        default_value_t = DEFAULT_HOTWORDS_SCORE,
        global = true,
        help = "指定 hotwords 加权分数",
        long_help = "指定 sherpa-onnx hotwords 加权分数. 仅支持 contextual biasing 的模型会使用该分数, 当前默认 Paraformer 模型使用词表后处理."
    )]
    pub hotwords_score: f32,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    #[command(
        alias = "d",
        alias = "listen",
        alias = "l",
        about = "启动热键监听和悬浮窗"
    )]
    Daemon(DaemonArgs),
    #[command(alias = "t", about = "转写音频文件或生成 SRT 字幕")]
    Transcribe(TranscribeArgs),
    #[command(alias = "m", about = "下载和检查本地模型")]
    Models(ModelsCommand),
    #[command(about = "列出可用麦克风输入设备")]
    Devices,
    #[command(alias = "cfg", about = "生成和检查配置文件")]
    Config(ConfigCommand),
    #[command(about = "生成和检查用户词汇表")]
    Dict(DictCommand),
    #[command(alias = "comp", about = "生成 shell 补全脚本")]
    Completion(CompletionArgs),
}

#[derive(Args, Debug)]
pub struct DaemonArgs {
    #[arg(
        long,
        default_value = "F2",
        env = "VOCOTYPE_HOTKEY",
        help = "按住录音的全局热键",
        long_help = "按住录音的全局热键. 支持单键或组合键. 组合键用 + 连接, 修饰键在前, 主键在后, 例如 F2, ctrl+f2, cmdorctrl+space."
    )]
    pub hotkey: String,

    #[arg(
        long,
        default_value = "pressed",
        help = "选择热键触发模式",
        long_help = "选择热键触发模式. pressed 表示按住 hotkey 录音并在松开时停止. toggle 表示按一次 hotkey 开始, 再按一次停止. trigger-end 表示按 hotkey 开始, 按 end-hotkey 停止."
    )]
    pub hotkey_mode: String,

    #[arg(
        long,
        help = "trigger-end 模式下用于停止录音的结束热键",
        long_help = "trigger-end 模式下用于停止录音的结束热键. 写法和 hotkey 相同, 例如 F3, ctrl+f3, cmdorctrl+space."
    )]
    pub end_hotkey: Option<String>,

    #[arg(long, default_value_t = false, help = "保存转写音频和结果用于数据集")]
    pub save_dataset: bool,

    #[arg(long, env = "VOCOTYPE_DATASET_DIR", help = "指定数据集保存目录")]
    pub dataset_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false, help = "注入文本后追加换行")]
    pub append_newline: bool,

    #[arg(long, default_value_t = false, help = "注入文本前删除末尾句号")]
    pub strip_trailing_period: bool,

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

    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "json",
        help = "等效于 --format srt"
    )]
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
pub struct ConfigCommand {
    #[command(subcommand)]
    pub command: ConfigSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ConfigSubcommand {
    #[command(about = "输出默认配置文件模板")]
    Default,
    #[command(about = "输出配置文件 JSON Schema")]
    Schema,
    #[command(about = "检查配置文件是否加载并生效")]
    Doctor,
}

#[derive(Args, Debug)]
pub struct DictCommand {
    #[command(subcommand)]
    pub command: DictSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum DictSubcommand {
    #[command(about = "输出默认词汇表模板")]
    Default,
    #[command(about = "检查词汇表是否加载并生效")]
    Doctor,
}

#[derive(Args, Debug)]
pub struct CompletionArgs {
    #[arg(value_enum, help = "要生成补全脚本的 shell")]
    pub shell: Shell,
}

pub async fn run() -> Result<()> {
    let matches = if should_run_app_daemon() {
        Cli::command().get_matches_from(["vocotype", "daemon"])
    } else {
        Cli::command().get_matches()
    };
    let mut cli = Cli::from_arg_matches(&matches)?;
    let config_path = cli.config.clone();
    if matches!(&cli.command, Command::Config(_)) {
        let Command::Config(args) = cli.command else {
            unreachable!();
        };
        return run_config_command(args, config_path.as_deref(), &matches);
    }
    if matches!(
        &cli.command,
        Command::Dict(DictCommand {
            command: DictSubcommand::Default,
        })
    ) {
        return run_dict_command(
            DictCommand {
                command: DictSubcommand::Default,
            },
            None,
        );
    }

    let config = AppConfig::load(cli.config.as_deref())?;
    apply_config(&mut cli, &matches, &config)?;
    if matches!(&cli.command, Command::Dict(_)) {
        let dict_path = cli.dict.clone();
        let Command::Dict(args) = cli.command else {
            unreachable!();
        };
        return run_dict_command(args, dict_path.as_deref());
    }

    let options = ModelOptions {
        model_dir: cli.model_dir.clone(),
        model_cache_dir: cli.model_cache_dir.clone(),
        revision: cli.model_revision.clone(),
    };
    let store = ModelStore::new(&options);
    let asr_options = AsrOptions {
        dictionary: SpeechDictionary::load(cli.dict.as_deref())?,
        hotwords_score: cli.hotwords_score,
        english_punctuation: config
            .post_processing
            .english_punctuation
            .unwrap_or(false),
        strip_trailing_period: config
            .post_processing
            .strip_trailing_period
            .or(config.daemon.strip_trailing_period)
            .unwrap_or(false),
    };

    match cli.command {
        Command::Daemon(args) => {
            let mut asr_options = asr_options;
            asr_options.strip_trailing_period = args.strip_trailing_period;
            let daemon = DaemonOptions {
                hotkey: args.hotkey,
                hotkey_mode: HotkeyMode::parse(&args.hotkey_mode)?,
                end_hotkey: args.end_hotkey,
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
                asr_options,
            };
            run_daemon(store, daemon).await
        }
        Command::Transcribe(args) => {
            match args.resolved_format() {
                TranscribeFormat::Json => {
                    let engine = AsrEngine::load_with_options(store, asr_options)?;
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
                            asr_options,
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
        Command::Config(_) => unreachable!("config command is handled before config loading"),
        Command::Dict(_) => unreachable!("dict command is handled before model loading"),
        Command::Completion(args) => {
            let mut command = Cli::command();
            generate(args.shell, &mut command, "vocotype", &mut std::io::stdout());
            Ok(())
        }
    }
}

fn should_run_app_daemon() -> bool {
    std::env::args_os().len() == 1 && launched_from_macos_app_bundle()
}

#[cfg(target_os = "macos")]
fn launched_from_macos_app_bundle() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .is_some_and(|parent| {
            parent.file_name().is_some_and(|name| name == "MacOS")
                && parent
                    .parent()
                    .is_some_and(|contents| contents.file_name().is_some_and(|name| name == "Contents"))
                && parent
                    .parent()
                    .and_then(Path::parent)
                    .is_some_and(|bundle| bundle.extension().is_some_and(|ext| ext == "app"))
        })
}

#[cfg(not(target_os = "macos"))]
fn launched_from_macos_app_bundle() -> bool {
    false
}

fn apply_config(cli: &mut Cli, matches: &ArgMatches, config: &AppConfig) -> Result<()> {
    cli.model_dir = merge_option(
        cli.model_dir.take(),
        matches.value_source("model_dir"),
        config.model_dir.clone(),
    );
    cli.model_cache_dir = merge_option(
        cli.model_cache_dir.take(),
        matches.value_source("model_cache_dir"),
        config.model_cache_dir.clone(),
    );
    cli.model_revision = merge_value(
        std::mem::take(&mut cli.model_revision),
        matches.value_source("model_revision"),
        config.model_revision.clone(),
    );
    cli.dict = merge_option(
        cli.dict.take(),
        matches.value_source("dict"),
        config.dict_path.clone(),
    );
    cli.hotwords_score = merge_value(
        cli.hotwords_score,
        matches.value_source("hotwords_score"),
        config.asr.hotwords_score,
    );

    if let Some((_, sub_matches)) = matches.subcommand() {
        match &mut cli.command {
            Command::Daemon(args) => apply_daemon_config(args, sub_matches, config),
            Command::Transcribe(args) => apply_transcribe_config(args, sub_matches, config)?,
            Command::Models(_)
            | Command::Devices
            | Command::Config(_)
            | Command::Dict(_)
            | Command::Completion(_) => {
            }
        }
    }

    Ok(())
}

fn run_config_command(
    args: ConfigCommand,
    config_path: Option<&Path>,
    matches: &ArgMatches,
) -> Result<()> {
    match args.command {
        ConfigSubcommand::Default => {
            print!("{}", default_config_template());
            Ok(())
        }
        ConfigSubcommand::Schema => {
            print!("{}", config_schema());
            Ok(())
        }
        ConfigSubcommand::Doctor => write_config_doctor(config_path, matches, io::stdout()),
    }
}

fn run_dict_command(args: DictCommand, dict_path: Option<&Path>) -> Result<()> {
    match args.command {
        DictSubcommand::Default => {
            print!("{}", default_dict_template());
            Ok(())
        }
        DictSubcommand::Doctor => write_dict_doctor(dict_path, io::stdout()),
    }
}

fn write_config_doctor(
    config_path: Option<&Path>,
    matches: &ArgMatches,
    mut writer: impl Write,
) -> Result<()> {
    let requested_path = config_path
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_path);
    writeln!(writer, "配置文件检查")?;
    writeln!(writer, "路径: {}", requested_path.display())?;
    writeln!(
        writer,
        "路径来源: {}",
        config_path_source_label(matches.value_source("config"))
    )?;

    let loaded = match AppConfig::load_with_metadata(config_path) {
        Ok(loaded) => loaded,
        Err(error) => {
            let message = format!("{:#}", error);
            writeln!(writer, "状态: 失败")?;
            writeln!(writer, "错误: {}", message)?;
            return Err(error);
        }
    };

    if loaded.found {
        writeln!(writer, "状态: 已加载")?;
    } else {
        writeln!(writer, "状态: 未找到, 当前会使用内置默认值")?;
    }
    writeln!(writer, "解析: 成功")?;
    writeln!(writer)?;
    writeln!(writer, "全局配置")?;
    write_global_config_status(&mut writer, matches, &loaded.config)?;
    writeln!(writer)?;
    writeln!(writer, "asr 配置")?;
    write_asr_config_status(&mut writer, matches, &loaded.config)?;
    writeln!(writer)?;
    writeln!(writer, "post-processing 配置")?;
    write_post_processing_config_status(&mut writer, &loaded.config)?;
    writeln!(writer)?;
    writeln!(writer, "daemon 配置")?;
    write_daemon_config_status(&mut writer, &loaded.config)?;
    writeln!(writer)?;
    writeln!(writer, "transcribe 配置")?;
    write_transcribe_config_status(&mut writer, &loaded.config)?;
    writeln!(writer)?;
    writeln!(
        writer,
        "说明: 运行具体子命令时, 该子命令的命令行参数仍会覆盖配置文件."
    )?;
    Ok(())
}

fn write_global_config_status(
    writer: &mut impl Write,
    matches: &ArgMatches,
    config: &AppConfig,
) -> Result<()> {
    write_value_status(
        writer,
        "model-dir",
        config
            .model_dir
            .as_ref()
            .map(|value| value.display().to_string()),
        matches.value_source("model_dir"),
    )?;
    write_value_status(
        writer,
        "model-cache-dir",
        config
            .model_cache_dir
            .as_ref()
            .map(|value| value.display().to_string()),
        matches.value_source("model_cache_dir"),
    )?;
    write_value_status(
        writer,
        "model-revision",
        config.model_revision.clone(),
        matches.value_source("model_revision"),
    )?;
    write_value_status(
        writer,
        "dict-path",
        config
            .dict_path
            .as_ref()
            .map(|value| value.display().to_string()),
        matches.value_source("dict"),
    )?;
    Ok(())
}

fn write_asr_config_status(
    writer: &mut impl Write,
    matches: &ArgMatches,
    config: &AppConfig,
) -> Result<()> {
    write_value_status(
        writer,
        "hotwords-score",
        config.asr.hotwords_score,
        matches.value_source("hotwords_score"),
    )?;
    Ok(())
}

fn write_post_processing_config_status(
    writer: &mut impl Write,
    config: &AppConfig,
) -> Result<()> {
    write_value_status_without_source(
        writer,
        "english-punctuation",
        config.post_processing.english_punctuation,
    )?;
    write_value_status_without_source(
        writer,
        "strip-trailing-period",
        config
            .post_processing
            .strip_trailing_period
            .or(config.daemon.strip_trailing_period),
    )?;
    Ok(())
}

fn write_daemon_config_status(writer: &mut impl Write, config: &AppConfig) -> Result<()> {
    let daemon = &config.daemon;
    write_env_value_status(writer, "hotkey", daemon.hotkey.clone(), "VOCOTYPE_HOTKEY")?;
    write_value_status_without_source(writer, "hotkey-mode", daemon.hotkey_mode.clone())?;
    write_value_status_without_source(writer, "end-hotkey", daemon.end_hotkey.clone())?;
    write_value_status_without_source(writer, "save-dataset", daemon.save_dataset)?;
    write_env_value_status(
        writer,
        "dataset-dir",
        daemon
            .dataset_dir
            .as_ref()
            .map(|value| value.display().to_string()),
        "VOCOTYPE_DATASET_DIR",
    )?;
    write_value_status_without_source(writer, "append-newline", daemon.append_newline)?;
    write_env_value_status(
        writer,
        "inject-method",
        daemon.inject_method.clone(),
        "VOCOTYPE_INJECT_METHOD",
    )?;
    write_value_status_without_source(writer, "end-silence-ms", daemon.end_silence_ms)?;
    write_value_status_without_source(writer, "pre-roll-ms", daemon.pre_roll_ms)?;
    write_value_status_without_source(writer, "tail-padding-ms", daemon.tail_padding_ms)?;
    write_value_status_without_source(writer, "min-speech-ms", daemon.min_speech_ms)?;
    write_value_status_without_source(writer, "max-segment-ms", daemon.max_segment_ms)?;
    write_value_status_without_source(writer, "idle-unload-secs", daemon.idle_unload_secs)?;
    Ok(())
}

fn write_transcribe_config_status(writer: &mut impl Write, config: &AppConfig) -> Result<()> {
    let transcribe = &config.transcribe;
    write_value_status_without_source(writer, "format", transcribe.format.clone())?;
    write_value_status_without_source(writer, "pretty", transcribe.pretty)?;
    write_value_status_without_source(writer, "subtitle-max-chars", transcribe.subtitle_max_chars)?;
    Ok(())
}

fn write_value_status<T: ToString>(
    writer: &mut impl Write,
    name: &str,
    value: Option<T>,
    source: Option<ValueSource>,
) -> Result<()> {
    let configured = value.as_ref().map(ToString::to_string);
    writeln!(
        writer,
        "- {}: {}, {}",
        name,
        configured.as_deref().unwrap_or("<unset>"),
        source_status_label(source, configured.is_some())
    )?;
    Ok(())
}

fn write_env_value_status<T: ToString>(
    writer: &mut impl Write,
    name: &str,
    value: Option<T>,
    env_key: &str,
) -> Result<()> {
    let configured = value.as_ref().map(ToString::to_string);
    let status = if std::env::var_os(env_key).is_some() {
        "运行对应子命令时会被环境变量覆盖"
    } else if configured.is_some() {
        "配置生效"
    } else {
        "使用内置默认值"
    };
    writeln!(
        writer,
        "- {}: {}, {}",
        name,
        configured.as_deref().unwrap_or("<unset>"),
        status
    )?;
    Ok(())
}

fn write_value_status_without_source<T: ToString>(
    writer: &mut impl Write,
    name: &str,
    value: Option<T>,
) -> Result<()> {
    let configured = value.as_ref().map(ToString::to_string);
    let status = if configured.is_some() {
        "配置生效"
    } else {
        "使用内置默认值"
    };
    writeln!(
        writer,
        "- {}: {}, {}",
        name,
        configured.as_deref().unwrap_or("<unset>"),
        status
    )?;
    Ok(())
}

fn config_path_source_label(source: Option<ValueSource>) -> &'static str {
    match source {
        Some(ValueSource::CommandLine) => "命令行",
        Some(ValueSource::EnvVariable) => "环境变量",
        _ => "默认路径",
    }
}

fn source_status_label(source: Option<ValueSource>, configured: bool) -> &'static str {
    match source {
        Some(ValueSource::CommandLine) => "被命令行覆盖",
        Some(ValueSource::EnvVariable) => "被环境变量覆盖",
        _ if configured => "配置生效",
        _ => "使用内置默认值",
    }
}

fn apply_daemon_config(args: &mut DaemonArgs, matches: &ArgMatches, config: &AppConfig) {
    let daemon = &config.daemon;
    args.hotkey = merge_value(
        std::mem::take(&mut args.hotkey),
        matches.value_source("hotkey"),
        daemon.hotkey.clone(),
    );
    args.hotkey_mode = merge_value(
        std::mem::take(&mut args.hotkey_mode),
        matches.value_source("hotkey_mode"),
        daemon.hotkey_mode.clone(),
    );
    args.end_hotkey = merge_option(
        args.end_hotkey.take(),
        matches.value_source("end_hotkey"),
        daemon.end_hotkey.clone(),
    );
    args.save_dataset = merge_value(
        args.save_dataset,
        matches.value_source("save_dataset"),
        daemon.save_dataset,
    );
    args.dataset_dir = merge_option(
        args.dataset_dir.take(),
        matches.value_source("dataset_dir"),
        daemon.dataset_dir.clone(),
    );
    args.append_newline = merge_value(
        args.append_newline,
        matches.value_source("append_newline"),
        daemon.append_newline,
    );
    args.strip_trailing_period = merge_value(
        args.strip_trailing_period,
        matches.value_source("strip_trailing_period"),
        config
            .post_processing
            .strip_trailing_period
            .or(daemon.strip_trailing_period),
    );
    args.inject_method = merge_value(
        std::mem::take(&mut args.inject_method),
        matches.value_source("inject_method"),
        daemon.inject_method.clone(),
    );
    args.end_silence_ms = merge_value(
        args.end_silence_ms,
        matches.value_source("end_silence_ms"),
        daemon.end_silence_ms,
    );
    args.pre_roll_ms = merge_value(
        args.pre_roll_ms,
        matches.value_source("pre_roll_ms"),
        daemon.pre_roll_ms,
    );
    args.tail_padding_ms = merge_value(
        args.tail_padding_ms,
        matches.value_source("tail_padding_ms"),
        daemon.tail_padding_ms,
    );
    args.min_speech_ms = merge_value(
        args.min_speech_ms,
        matches.value_source("min_speech_ms"),
        daemon.min_speech_ms,
    );
    args.max_segment_ms = merge_value(
        args.max_segment_ms,
        matches.value_source("max_segment_ms"),
        daemon.max_segment_ms,
    );
    args.idle_unload_secs = merge_value(
        args.idle_unload_secs,
        matches.value_source("idle_unload_secs"),
        daemon.idle_unload_secs,
    );
}

fn apply_transcribe_config(
    args: &mut TranscribeArgs,
    matches: &ArgMatches,
    config: &AppConfig,
) -> Result<()> {
    let transcribe = &config.transcribe;
    let configured_format = transcribe
        .format
        .as_deref()
        .map(parse_transcribe_format)
        .transpose()?;
    args.format = merge_value(
        args.format.clone(),
        matches.value_source("format"),
        configured_format,
    );
    args.pretty = merge_value(
        args.pretty,
        matches.value_source("pretty"),
        transcribe.pretty,
    );
    args.subtitle_max_chars = merge_value(
        args.subtitle_max_chars,
        matches.value_source("subtitle_max_chars"),
        transcribe.subtitle_max_chars,
    );
    Ok(())
}

fn parse_transcribe_format(value: &str) -> Result<TranscribeFormat> {
    TranscribeFormat::from_str(value, true)
        .map_err(|error| anyhow!("不支持的转写输出格式: {}", error))
}

fn merge_value<T>(value: T, source: Option<ValueSource>, configured: Option<T>) -> T {
    if should_use_config(source) {
        configured.unwrap_or(value)
    } else {
        value
    }
}

fn merge_option<T>(
    value: Option<T>,
    source: Option<ValueSource>,
    configured: Option<T>,
) -> Option<T> {
    if should_use_config(source) {
        configured.or(value)
    } else {
        value
    }
}

fn should_use_config(source: Option<ValueSource>) -> bool {
    matches!(source, None | Some(ValueSource::DefaultValue))
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use crate::config::{DaemonConfig, PostProcessingConfig, TranscribeConfig};

    #[test]
    fn config_default_prints_template() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "config", "default"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();

        let Command::Config(args) = cli.command else {
            panic!("expected config command");
        };
        assert!(matches!(args.command, ConfigSubcommand::Default));
        AppConfig::from_toml(default_config_template()).unwrap();
    }

    #[test]
    fn config_schema_prints_json_schema() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "config", "schema"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();

        let Command::Config(args) = cli.command else {
            panic!("expected config command");
        };
        assert!(matches!(args.command, ConfigSubcommand::Schema));
        serde_json::from_str::<serde_json::Value>(config_schema()).unwrap();
    }

    #[test]
    fn config_doctor_reports_missing_default_config() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "config", "doctor"])
            .unwrap();
        let mut output = Vec::new();

        write_config_doctor(None, &matches, &mut output).unwrap();

        let report = String::from_utf8(output).unwrap();
        assert!(report.contains("状态:"));
        assert!(report.contains("解析: 成功"));
    }

    #[test]
    fn config_fills_daemon_defaults() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "daemon"])
            .unwrap();
        let mut cli = Cli::from_arg_matches(&matches).unwrap();
        let config = AppConfig {
            daemon: DaemonConfig {
                hotkey: Some("F3".to_string()),
                hotkey_mode: Some("trigger-end".to_string()),
                end_hotkey: Some("F4".to_string()),
                append_newline: Some(true),
                idle_unload_secs: Some(0),
                ..Default::default()
            },
            post_processing: PostProcessingConfig {
                strip_trailing_period: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };

        apply_config(&mut cli, &matches, &config).unwrap();

        let Command::Daemon(args) = cli.command else {
            panic!("expected daemon command");
        };
        assert_eq!(args.hotkey, "F3");
        assert_eq!(args.hotkey_mode, "trigger-end");
        assert_eq!(args.end_hotkey.as_deref(), Some("F4"));
        assert!(args.append_newline);
        assert!(args.strip_trailing_period);
        assert_eq!(args.idle_unload_secs, 0);
    }

    #[test]
    fn command_line_global_values_override_config_after_subcommand() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "daemon", "--model-dir", "/tmp/cli-models"])
            .unwrap();
        let mut cli = Cli::from_arg_matches(&matches).unwrap();
        let config = AppConfig {
            model_dir: Some(PathBuf::from("/tmp/config-models")),
            ..Default::default()
        };

        apply_config(&mut cli, &matches, &config).unwrap();

        assert_eq!(cli.model_dir, Some(PathBuf::from("/tmp/cli-models")));
    }

    #[test]
    fn command_line_daemon_values_override_config() {
        let matches = Cli::command()
            .try_get_matches_from([
                "vocotype",
                "daemon",
                "--hotkey",
                "F4",
                "--idle-unload-secs",
                "12",
            ])
            .unwrap();
        let mut cli = Cli::from_arg_matches(&matches).unwrap();
        let config = AppConfig {
            daemon: DaemonConfig {
                hotkey: Some("F3".to_string()),
                idle_unload_secs: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };

        apply_config(&mut cli, &matches, &config).unwrap();

        let Command::Daemon(args) = cli.command else {
            panic!("expected daemon command");
        };
        assert_eq!(args.hotkey, "F4");
        assert_eq!(args.idle_unload_secs, 12);
    }

    #[test]
    fn config_fills_transcribe_defaults() {
        let matches = Cli::command()
            .try_get_matches_from(["vocotype", "transcribe", "sample.wav"])
            .unwrap();
        let mut cli = Cli::from_arg_matches(&matches).unwrap();
        let config = AppConfig {
            transcribe: TranscribeConfig {
                format: Some("srt".to_string()),
                pretty: Some(true),
                subtitle_max_chars: Some(40),
            },
            ..Default::default()
        };

        apply_config(&mut cli, &matches, &config).unwrap();

        let Command::Transcribe(args) = cli.command else {
            panic!("expected transcribe command");
        };
        assert!(matches!(args.resolved_format(), TranscribeFormat::Srt));
        assert!(args.pretty);
        assert_eq!(args.subtitle_max_chars, 40);
    }
}
