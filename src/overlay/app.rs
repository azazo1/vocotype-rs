use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use super::state::{OverlayMode, OverlayState};

const REPAINT_INTERVAL: Duration = Duration::from_millis(80);
const DONE_VISIBLE_FOR: Duration = Duration::from_millis(1_400);
const MIN_WIDTH: f32 = 360.0;
const MAX_WIDTH: f32 = 560.0;
const BASE_HEIGHT: f32 = 108.0;
const MAX_HEIGHT: f32 = 420.0;
const TEXT_LINE_HEIGHT: f32 = 21.0;
const TEXT_CHARS_PER_LINE: usize = 32;

pub(crate) struct OverlayApp {
    state: Arc<Mutex<OverlayState>>,
    receiver: Receiver<OverlayState>,
    visible: bool,
    initial_hide_sent: bool,
    hide_at: Option<Instant>,
}

impl OverlayApp {
    pub(crate) fn new(state: Arc<Mutex<OverlayState>>, receiver: Receiver<OverlayState>) -> Self {
        Self {
            state,
            receiver,
            visible: false,
            initial_hide_sent: false,
            hide_at: None,
        }
    }

    fn ensure_initial_hidden(&mut self, ctx: &egui::Context) {
        if !self.initial_hide_sent {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.initial_hide_sent = true;
        }
    }

    fn drain_updates(&mut self, ctx: &egui::Context) {
        while let Ok(state) = self.receiver.try_recv() {
            self.apply_visibility(ctx, &state);
            if let Ok(mut guard) = self.state.lock() {
                *guard = state;
            }
        }
    }

    fn apply_visibility(&mut self, ctx: &egui::Context, state: &OverlayState) {
        if state.is_visible() {
            self.resize(ctx, state);
            self.show(ctx);
            self.hide_at = if state.is_done() {
                Some(Instant::now() + DONE_VISIBLE_FOR)
            } else {
                None
            };
        } else {
            self.hide(ctx);
            self.hide_at = None;
        }
    }

    fn handle_auto_hide(&mut self, ctx: &egui::Context) {
        let Some(hide_at) = self.hide_at else {
            return;
        };
        if Instant::now() >= hide_at {
            self.hide(ctx);
            self.clear_state();
            self.hide_at = None;
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
            self.visible = true;
        }
    }

    fn hide(&mut self, ctx: &egui::Context) {
        if self.visible || !self.initial_hide_sent {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.visible = false;
            self.initial_hide_sent = true;
        }
    }

    fn resize(&self, ctx: &egui::Context, state: &OverlayState) {
        let size = overlay_size(state);
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
    }

    fn current_state(&self) -> OverlayState {
        self.state
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn clear_state(&self) {
        if let Ok(mut guard) = self.state.lock() {
            *guard = OverlayState::default();
        }
    }
}

impl eframe::App for OverlayApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_initial_hidden(ctx);
        self.drain_updates(ctx);
        self.handle_auto_hide(ctx);
        ctx.request_repaint_after(REPAINT_INTERVAL);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.visible {
            draw_overlay(ui, &self.current_state());
        }
    }
}

fn draw_overlay(ui: &mut egui::Ui, state: &OverlayState) {
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(egui::Color32::from_rgb(24, 28, 33)))
        .show(ui, |ui| {
            ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    draw_status(ui, state);
                    draw_transcript(ui, state);
                });
            });
        });
}

fn draw_status(ui: &mut egui::Ui, state: &OverlayState) {
    ui.heading(state.mode.title());
    ui.add(
        egui::Label::new(egui::RichText::new(state.mode.detail()).size(14.0))
            .wrap(),
    );
    if let OverlayMode::Recording { level } = state.mode {
        ui.add(
            egui::ProgressBar::new(level.clamp(0.0, 1.0)).show_percentage(),
        );
    }
}

fn draw_transcript(ui: &mut egui::Ui, state: &OverlayState) {
    if state.transcript_lines.is_empty() {
        return;
    }

    ui.add_space(10.0);
    for line in &state.transcript_lines {
        ui.add(
            egui::Label::new(
                egui::RichText::new(line)
                    .size(15.0)
                    .color(egui::Color32::from_rgb(235, 238, 242)),
            )
            .wrap(),
        );
    }
}

fn overlay_size(state: &OverlayState) -> egui::Vec2 {
    let wrapped_lines = state
        .transcript_lines
        .iter()
        .map(|line| visual_line_count(line))
        .sum::<usize>();
    let text_height = if wrapped_lines == 0 {
        0.0
    } else {
        18.0 + wrapped_lines as f32 * TEXT_LINE_HEIGHT
    };
    let width = if wrapped_lines == 0 {
        MIN_WIDTH
    } else {
        MAX_WIDTH
    };
    egui::vec2(width, (BASE_HEIGHT + text_height).min(MAX_HEIGHT))
}

fn visual_line_count(text: &str) -> usize {
    let chars = text.chars().count().max(1);
    chars.div_ceil(TEXT_CHARS_PER_LINE)
}
