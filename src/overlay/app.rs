use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use super::{OVERLAY_WIDTH, platform};
use super::state::{OverlayMode, OverlayState};

const REPAINT_INTERVAL: Duration = Duration::from_millis(80);
const DONE_VISIBLE_FOR: Duration = Duration::from_millis(1_400);
const ERROR_HIDE_AFTER_LEAVE: Duration = Duration::from_millis(700);
const BASE_HEIGHT: f32 = 108.0;
const MAX_HEIGHT: f32 = 420.0;
const TEXT_LINE_HEIGHT: f32 = 21.0;
const TEXT_CHARS_PER_LINE: usize = 32;
const SCREEN_MARGIN: f32 = 24.0;

pub(crate) struct OverlayApp {
    state: Arc<Mutex<OverlayState>>,
    receiver: Receiver<OverlayState>,
    visible: bool,
    _status_item: Option<platform::StatusItem>,
    initial_hide_sent: bool,
    hide_at: Option<Instant>,
    mouse_passthrough: bool,
    error_hovered: bool,
    error_was_hovered: bool,
}

impl OverlayApp {
    pub(crate) fn new(
        state: Arc<Mutex<OverlayState>>,
        receiver: Receiver<OverlayState>,
        status_item: Option<platform::StatusItem>,
    ) -> Self {
        Self {
            state,
            receiver,
            visible: false,
            _status_item: status_item,
            initial_hide_sent: false,
            hide_at: None,
            mouse_passthrough: true,
            error_hovered: false,
            error_was_hovered: false,
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
            let size = overlay_size(state);
            self.resize(ctx, size);
            self.reposition(ctx, size);
            self.show(ctx);
            self.set_mouse_passthrough(ctx, !state.is_error());
            self.hide_at = if state.is_done() {
                Some(Instant::now() + DONE_VISIBLE_FOR)
            } else {
                None
            };
            if !state.is_error() {
                self.error_hovered = false;
                self.error_was_hovered = false;
            }
        } else {
            self.hide(ctx);
            self.hide_at = None;
            self.set_mouse_passthrough(ctx, true);
            self.error_hovered = false;
            self.error_was_hovered = false;
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

    fn handle_error_hover(&mut self, ctx: &egui::Context) {
        if !self.visible || !self.current_state().is_error() {
            return;
        }

        let hovered = ctx.pointer_hover_pos().is_some();
        match (self.error_hovered, hovered) {
            (false, true) => {
                self.error_was_hovered = true;
                self.hide_at = None;
            }
            (true, false) if self.error_was_hovered => {
                self.hide_at = Some(Instant::now() + ERROR_HIDE_AFTER_LEAVE);
            }
            _ => {}
        }
        self.error_hovered = hovered;
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

    fn resize(&self, ctx: &egui::Context, size: egui::Vec2) {
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
    }

    fn reposition(&self, ctx: &egui::Context, size: egui::Vec2) {
        if let Some(screen_rect) = platform::current_mouse_screen_rect() {
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(
                overlay_position(screen_rect, size),
            ));
        }
    }

    fn set_mouse_passthrough(&mut self, ctx: &egui::Context, passthrough: bool) {
        if self.mouse_passthrough != passthrough {
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(passthrough));
            self.mouse_passthrough = passthrough;
        }
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
        self.handle_error_hover(ctx);
        self.handle_auto_hide(ctx);
        ctx.request_repaint_after(REPAINT_INTERVAL);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.visible {
            draw_overlay(ui, &self.current_state(), self.error_hovered);
        }
    }
}

fn draw_overlay(ui: &mut egui::Ui, state: &OverlayState, error_hovered: bool) {
    let fill = overlay_fill(state, error_hovered);
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(fill))
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

fn overlay_fill(state: &OverlayState, error_hovered: bool) -> egui::Color32 {
    if state.is_error() && error_hovered {
        egui::Color32::from_rgb(88, 35, 35)
    } else if state.is_error() {
        egui::Color32::from_rgb(52, 35, 35)
    } else {
        egui::Color32::from_rgb(24, 28, 33)
    }
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
    if state.transcript_lines.is_empty() && state.streaming.is_none() {
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
    if let Some(streaming) = &state.streaming {
        ui.horizontal_wrapped(|ui| {
            if !streaming.stable.is_empty() {
                ui.label(
                    egui::RichText::new(&streaming.stable)
                        .size(15.0)
                        .color(egui::Color32::from_rgb(235, 238, 242)),
                );
            }
            if !streaming.unstable.is_empty() {
                ui.label(
                    egui::RichText::new(&streaming.unstable)
                        .size(15.0)
                        .color(egui::Color32::from_rgb(145, 154, 166)),
                );
            }
        });
    }
}

fn overlay_size(state: &OverlayState) -> egui::Vec2 {
    let wrapped_lines = state
        .transcript_lines
        .iter()
        .map(|line| visual_line_count(line))
        .sum::<usize>();
    let streaming_lines = state
        .streaming
        .as_ref()
        .map(|streaming| {
            visual_line_count(&format!("{}{}", streaming.stable, streaming.unstable))
        })
        .unwrap_or(0);
    let wrapped_lines = wrapped_lines + streaming_lines;
    let text_height = if wrapped_lines == 0 {
        0.0
    } else {
        18.0 + wrapped_lines as f32 * TEXT_LINE_HEIGHT
    };
    egui::vec2(
        OVERLAY_WIDTH,
        (BASE_HEIGHT + text_height).min(MAX_HEIGHT),
    )
}

fn overlay_position(screen_rect: platform::ScreenRect, size: egui::Vec2) -> egui::Pos2 {
    let available = (screen_rect.size - size).max(egui::Vec2::ZERO);
    let margin = SCREEN_MARGIN.min(available.x).min(available.y);
    let x = screen_rect.min.x + available.x - margin;
    let y = screen_rect.min.y + margin;
    egui::pos2(x, y)
}

fn visual_line_count(text: &str) -> usize {
    let chars = text.chars().count().max(1);
    chars.div_ceil(TEXT_CHARS_PER_LINE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::StreamingTranscript;

    #[test]
    fn overlay_width_stays_fixed_when_streaming_text_changes() {
        let idle = OverlayState::new(OverlayMode::Recording { level: 0.0 });
        let streaming = OverlayState::with_streaming(
            OverlayMode::Streaming { revision: false },
            Vec::new(),
            StreamingTranscript {
                stable: "这是一个".to_string(),
                unstable: "实时流式转写结果".to_string(),
            },
        );

        assert_eq!(overlay_size(&idle).x, OVERLAY_WIDTH);
        assert_eq!(overlay_size(&streaming).x, OVERLAY_WIDTH);
    }
}
