//! 悬浮窗

mod app;
mod fonts;
mod platform;
mod state;

pub(super) const OVERLAY_WIDTH: f32 = 560.0;

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use tracing::{info, warn};

use app::OverlayApp;

pub use state::{OverlayMode, OverlayState, StreamingTranscript};

#[derive(Clone)]
pub struct OverlayHandle {
    sender: Sender<OverlayState>,
}

pub struct OverlayRunner {
    state: Arc<Mutex<OverlayState>>,
    receiver: Receiver<OverlayState>,
}

impl OverlayHandle {
    pub fn set(&self, state: OverlayState) {
        let _ = self.sender.send(state);
    }

    pub fn idle(&self) {
        self.set(OverlayState::new(OverlayMode::Idle));
    }
}

pub fn create() -> (OverlayHandle, OverlayRunner) {
    let (sender, receiver) = unbounded::<OverlayState>();
    let state = Arc::new(Mutex::new(OverlayState::default()));
    (
        OverlayHandle { sender },
        OverlayRunner { state, receiver },
    )
}

impl OverlayRunner {
    pub fn run(self) -> Result<()> {
        let mut native_options = eframe::NativeOptions {
            viewport: overlay_viewport(),
            ..Default::default()
        };
        platform::configure_native_options(&mut native_options);

        let app_state = self.state;
        let receiver = self.receiver;
        info!("悬浮窗事件循环已启动");
        eframe::run_native(
            "VocoType",
            native_options,
            Box::new(move |cc| {
                fonts::install(&cc.egui_ctx);
                platform::configure_window(cc);
                let status_item = platform::install_status_item();
                Ok(Box::new(OverlayApp::new(app_state, receiver, status_item)))
            }),
        )
        .map_err(|error| {
            warn!(%error, "悬浮窗退出");
            anyhow!("悬浮窗运行失败: {}", error)
        })
    }
}

fn overlay_viewport() -> egui::ViewportBuilder {
    egui::ViewportBuilder::default()
        .with_title("VocoType")
        .with_decorations(false)
        .with_resizable(false)
        .with_inner_size([OVERLAY_WIDTH, 108.0])
        .with_always_on_top()
        .with_visible(false)
        .with_active(false)
        .with_taskbar(false)
        .with_mouse_passthrough(true)
}
