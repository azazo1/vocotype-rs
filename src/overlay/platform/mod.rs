#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "windows")]
mod windows;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenRect {
    pub(crate) min: egui::Pos2,
    pub(crate) size: egui::Vec2,
}

#[cfg(target_os = "macos")]
pub(crate) use macos::{
    configure_native_options, configure_window, current_mouse_screen_rect,
};
#[cfg(target_os = "linux")]
pub(crate) use linux::current_mouse_screen_rect;
#[cfg(target_os = "windows")]
pub(crate) use windows::current_mouse_screen_rect;

#[cfg(not(target_os = "macos"))]
pub(crate) fn configure_native_options(_native_options: &mut eframe::NativeOptions) {}

#[cfg(not(target_os = "macos"))]
pub(crate) fn configure_window(_cc: &eframe::CreationContext<'_>) {}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows"
)))]
pub(crate) fn current_mouse_screen_rect() -> Option<ScreenRect> {
    None
}
