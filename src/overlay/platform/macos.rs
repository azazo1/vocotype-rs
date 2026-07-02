use tracing::warn;

use super::ScreenRect;

pub(crate) fn configure_native_options(native_options: &mut eframe::NativeOptions) {
    native_options.event_loop_builder = Some(Box::new(|builder| {
        use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};

        builder.with_activation_policy(ActivationPolicy::Accessory);
    }));
}

pub(crate) fn configure_window(cc: &eframe::CreationContext<'_>) {
    if let Err(error) = configure_window_inner(cc) {
        warn!(%error, "无法应用 macOS 悬浮窗属性");
    }
}

pub(crate) fn current_mouse_screen_rect() -> Option<ScreenRect> {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSEvent, NSScreen};
    use objc2_core_graphics::{CGDisplayBounds, CGMainDisplayID};

    let mtm = MainThreadMarker::new()?;
    let mouse = NSEvent::mouseLocation();
    let screens = NSScreen::screens(mtm);
    for index in 0..screens.len() {
        let screen = unsafe { screens.objectAtIndex_unchecked(index) };
        let frame = screen.frame();
        let frame_min = frame.origin;
        let frame_max = frame.max();
        if mouse.x < frame_min.x
            || mouse.x >= frame_max.x
            || mouse.y < frame_min.y
            || mouse.y >= frame_max.y
        {
            continue;
        }

        let visible_frame = screen.visibleFrame();
        let main_frame = CGDisplayBounds(CGMainDisplayID());
        let y =
            main_frame.size.height - visible_frame.size.height - visible_frame.origin.y;
        return Some(ScreenRect {
            min: egui::pos2(visible_frame.origin.x as f32, y as f32),
            size: egui::vec2(
                visible_frame.size.width as f32,
                visible_frame.size.height as f32,
            ),
        });
    }
    None
}

fn configure_window_inner(cc: &eframe::CreationContext<'_>) -> Result<(), String> {
    use objc2::rc::Retained;
    use objc2_app_kit::{NSView, NSWindowCollectionBehavior};
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let handle = cc.window_handle().map_err(|error| error.to_string())?.as_raw();
    let RawWindowHandle::AppKit(handle) = handle else {
        return Ok(());
    };

    let ns_view_ptr = handle.ns_view.as_ptr();
    let ns_view: Retained<NSView> =
        unsafe { Retained::retain(ns_view_ptr.cast()) }.ok_or("NSView 指针无效")?;
    let ns_window = ns_view.window().ok_or("NSView 未关联 NSWindow")?;
    let behavior = NSWindowCollectionBehavior::CanJoinAllSpaces
        | NSWindowCollectionBehavior::FullScreenAuxiliary
        | NSWindowCollectionBehavior::Transient
        | NSWindowCollectionBehavior::Stationary
        | NSWindowCollectionBehavior::IgnoresCycle;

    ns_window.setCollectionBehavior(behavior);
    Ok(())
}
