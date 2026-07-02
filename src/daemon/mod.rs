mod runtime;
mod segments;
mod state;
mod worker;

use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::error;

use crate::inject::InjectMethod;
use crate::models::ModelStore;
use crate::overlay::{OverlayMode, OverlayState, create as create_overlay};

#[derive(Clone, Debug)]
pub struct DaemonOptions {
    pub hotkey: String,
    pub save_dataset: bool,
    pub dataset_dir: Option<PathBuf>,
    pub append_newline: bool,
    pub inject_method: InjectMethod,
    pub end_silence_ms: u32,
    pub pre_roll_ms: u32,
    pub tail_padding_ms: u32,
    pub min_speech_ms: u32,
    pub max_segment_ms: u32,
    pub idle_unload_secs: u64,
}

pub async fn run_daemon(store: ModelStore, options: DaemonOptions) -> Result<()> {
    store.paths.ensure_dirs()?;
    if let Err(error) = store.verify_required() {
        error!(%error, "模型缺失");
        eprintln!("模型缺失, 请先运行: {}", store.download_hint());
        return Err(error);
    }

    let (overlay, overlay_runner) = create_overlay();
    overlay.idle();
    let daemon_overlay = overlay.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(error) = runtime::run_daemon_loop(store, options, daemon_overlay) {
            error!(%error, "daemon 运行失败");
            overlay.set(OverlayState::new(
                OverlayMode::Error {
                    message: error.to_string(),
                },
            ));
        }
    });

    overlay_runner.run().context("无法启动悬浮窗")
}
