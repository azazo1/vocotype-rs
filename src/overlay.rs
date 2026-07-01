use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use crossbeam_channel::{Sender, unbounded};
use tracing::{info, warn};

#[derive(Clone, Debug)]
pub enum OverlayMode {
    Idle,
    Recording { level: f32 },
    Silence { pending: usize },
    Transcribing { pending: usize },
    Done { text: String },
    Error { message: String },
}

#[derive(Clone, Debug)]
pub struct OverlayState {
    pub mode: OverlayMode,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            mode: OverlayMode::Idle,
        }
    }
}

#[derive(Clone)]
pub struct OverlayHandle {
    sender: Sender<OverlayState>,
}

impl OverlayHandle {
    pub fn set(&self, state: OverlayState) {
        let _ = self.sender.send(state);
    }

    pub fn idle(&self) {
        self.set(OverlayState {
            mode: OverlayMode::Idle,
        });
    }
}

pub fn start() -> Result<OverlayHandle> {
    let (sender, receiver) = unbounded::<OverlayState>();
    let state = Arc::new(Mutex::new(OverlayState::default()));
    let thread_state = state.clone();

    thread::spawn(move || {
        let app = OverlayApp {
            state: thread_state,
            receiver,
        };
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("VocoType")
                .with_decorations(false)
                .with_resizable(false)
                .with_inner_size([360.0, 108.0])
                .with_always_on_top(),
            ..Default::default()
        };
        if let Err(error) =
            eframe::run_native("VocoType", native_options, Box::new(|_| Ok(Box::new(app))))
        {
            warn!(%error, "悬浮窗退出");
        }
    });

    info!("悬浮窗线程已启动");
    Ok(OverlayHandle { sender })
}

struct OverlayApp {
    state: Arc<Mutex<OverlayState>>,
    receiver: crossbeam_channel::Receiver<OverlayState>,
}

impl eframe::App for OverlayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(state) = self.receiver.try_recv() {
            if let Ok(mut guard) = self.state.lock() {
                *guard = state;
            }
        }

        let state = self
            .state
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
            ui.painter()
                .rect_filled(ui.max_rect(), 8.0, egui::Color32::from_rgb(24, 28, 33));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.heading(mode_title(&state.mode));
                    ui.label(mode_detail(&state.mode));
                    if let OverlayMode::Recording { level } = state.mode {
                        ui.add(egui::ProgressBar::new(level.clamp(0.0, 1.0)).show_percentage());
                    }
                });
            });
        });
        ctx.request_repaint_after(std::time::Duration::from_millis(80));
    }
}

fn mode_title(mode: &OverlayMode) -> &'static str {
    match mode {
        OverlayMode::Idle => "VocoType",
        OverlayMode::Recording { .. } => "正在录音",
        OverlayMode::Silence { .. } => "等待语音",
        OverlayMode::Transcribing { .. } => "正在转写",
        OverlayMode::Done { .. } => "转写完成",
        OverlayMode::Error { .. } => "需要处理",
    }
}

fn mode_detail(mode: &OverlayMode) -> String {
    match mode {
        OverlayMode::Idle => "按住热键开始本地转写".to_string(),
        OverlayMode::Recording { .. } => "继续说话, 停顿后会自动提交当前片段".to_string(),
        OverlayMode::Silence { pending } => format!("检测到停顿, 队列中有 {} 个片段", pending),
        OverlayMode::Transcribing { pending } => format!("后台转写中, 队列剩余 {}", pending),
        OverlayMode::Done { text } => {
            if text.chars().count() > 36 {
                format!("{}...", text.chars().take(36).collect::<String>())
            } else {
                text.clone()
            }
        }
        OverlayMode::Error { message } => message.clone(),
    }
}
