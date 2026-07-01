use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::asr::AsrEngine;
use crate::audio::list_input_devices;
use crate::daemon::{DaemonOptions, run_daemon};
use crate::inject::InjectMethod;
use crate::models::{DEFAULT_REVISION, ModelOptions, ModelStore};

#[derive(Parser, Debug)]
#[command(name = "vocotype", version, about = "VocoType Rust 版")]
pub struct Cli {
    #[arg(long, env = "VOCOTYPE_MODEL_DIR", global = true)]
    pub model_dir: Option<PathBuf>,

    #[arg(long, env = "VOCOTYPE_MODEL_CACHE_DIR", global = true)]
    pub model_cache_dir: Option<PathBuf>,

    #[arg(long, env = "VOCOTYPE_MODEL_REVISION", default_value = DEFAULT_REVISION, global = true)]
    pub model_revision: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Daemon(DaemonArgs),
    Transcribe(TranscribeArgs),
    Models(ModelsCommand),
    Devices,
}

#[derive(Args, Debug)]
pub struct DaemonArgs {
    #[arg(long, default_value = "F2", env = "VOCOTYPE_HOTKEY")]
    pub hotkey: String,

    #[arg(long, default_value_t = false)]
    pub save_dataset: bool,

    #[arg(long, env = "VOCOTYPE_DATASET_DIR")]
    pub dataset_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub append_newline: bool,

    #[arg(long, env = "VOCOTYPE_INJECT_METHOD", default_value = "auto")]
    pub inject_method: String,

    #[arg(long, default_value_t = 650)]
    pub end_silence_ms: u32,

    #[arg(long, default_value_t = 180)]
    pub pre_roll_ms: u32,

    #[arg(long, default_value_t = 180)]
    pub tail_padding_ms: u32,

    #[arg(long, default_value_t = 240)]
    pub min_speech_ms: u32,

    #[arg(long, default_value_t = 15_000)]
    pub max_segment_ms: u32,
}

#[derive(Args, Debug)]
pub struct TranscribeArgs {
    #[arg(long)]
    pub audio: PathBuf,

    #[arg(long, default_value_t = false)]
    pub pretty: bool,
}

#[derive(Args, Debug)]
pub struct ModelsCommand {
    #[command(subcommand)]
    pub command: ModelsSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelsSubcommand {
    Download,
    Doctor,
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
            };
            run_daemon(store, daemon).await
        }
        Command::Transcribe(args) => {
            let engine = AsrEngine::load(store)?;
            let result = engine.transcribe_file(&args.audio)?;
            if args.pretty {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("{}", serde_json::to_string(&result)?);
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
    }
}
