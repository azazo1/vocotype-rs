#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub(crate) use macos::{configure_native_options, configure_window};

#[cfg(not(target_os = "macos"))]
pub(crate) fn configure_native_options(_native_options: &mut eframe::NativeOptions) {}

#[cfg(not(target_os = "macos"))]
pub(crate) fn configure_window(_cc: &eframe::CreationContext<'_>) {}
