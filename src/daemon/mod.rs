mod runtime;
mod segments;
mod state;
mod worker;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use tracing::error;

use crate::asr::AsrOptions;
use crate::hotkey::{HotkeyConfig, HotkeyManager};
use crate::inject::InjectMethod;
use crate::models::ModelStore;
use crate::overlay::{OverlayMode, OverlayState, create as create_overlay};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HotkeyMode {
    Pressed,
    Toggle,
    TriggerEnd,
}

impl HotkeyMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pressed" | "press" | "hold" => Ok(Self::Pressed),
            "toggle" => Ok(Self::Toggle),
            "trigger-end" | "trigger_end" | "triggerend" => Ok(Self::TriggerEnd),
            _ => anyhow::bail!("不支持的热键模式: {}", value),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pressed => "pressed",
            Self::Toggle => "toggle",
            Self::TriggerEnd => "trigger-end",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DaemonOptions {
    pub hotkey: String,
    pub hotkey_mode: HotkeyMode,
    pub end_hotkey: Option<String>,
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
    pub asr_options: AsrOptions,
}

pub async fn run_daemon(store: ModelStore, options: DaemonOptions) -> Result<()> {
    store.paths.ensure_dirs()?;
    if let Err(error) = store.verify_required_for(options.asr_options.backend) {
        error!(%error, "模型缺失");
        eprintln!(
            "模型缺失, 请先运行: {}",
            store.download_hint_for(options.asr_options.backend)
        );
        return Err(error);
    }

    let (overlay, overlay_runner) = create_overlay();
    overlay.idle();
    let hotkey_cfg = HotkeyConfig {
        key: options.hotkey.clone(),
        end_key: matches!(options.hotkey_mode, HotkeyMode::TriggerEnd)
            .then(|| options.end_hotkey.clone())
            .flatten(),
    };
    if matches!(options.hotkey_mode, HotkeyMode::TriggerEnd) && hotkey_cfg.end_key.is_none() {
        bail!("trigger-end 热键模式需要配置 end-hotkey");
    }
    let hotkey_manager = HotkeyManager::new(&hotkey_cfg)?;
    let hotkeys = hotkey_manager.matcher();
    let daemon_overlay = overlay.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(error) = runtime::run_daemon_loop(store, options, daemon_overlay, hotkeys) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hotkey_modes() {
        assert_eq!(HotkeyMode::parse("pressed").unwrap(), HotkeyMode::Pressed);
        assert_eq!(HotkeyMode::parse("hold").unwrap(), HotkeyMode::Pressed);
        assert_eq!(HotkeyMode::parse("toggle").unwrap(), HotkeyMode::Toggle);
        assert_eq!(
            HotkeyMode::parse("trigger_end").unwrap(),
            HotkeyMode::TriggerEnd
        );
    }

    #[test]
    fn rejects_unknown_hotkey_mode() {
        assert!(HotkeyMode::parse("single").is_err());
    }
}
