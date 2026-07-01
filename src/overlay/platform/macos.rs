use tracing::warn;

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

fn configure_window_inner(cc: &eframe::CreationContext<'_>) -> Result<(), String> {
    use objc2::rc::Id;
    use objc2_app_kit::{NSView, NSWindowCollectionBehavior};
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let handle = cc.window_handle().map_err(|error| error.to_string())?.as_raw();
    let RawWindowHandle::AppKit(handle) = handle else {
        return Ok(());
    };

    let ns_view_ptr = handle.ns_view.as_ptr();
    let ns_view: Id<NSView> =
        unsafe { Id::retain(ns_view_ptr.cast()) }.ok_or("NSView 指针无效")?;
    let ns_window = ns_view.window().ok_or("NSView 未关联 NSWindow")?;
    let behavior = NSWindowCollectionBehavior::CanJoinAllSpaces
        | NSWindowCollectionBehavior::FullScreenAuxiliary
        | NSWindowCollectionBehavior::Transient
        | NSWindowCollectionBehavior::Stationary
        | NSWindowCollectionBehavior::IgnoresCycle;

    unsafe {
        ns_window.setCollectionBehavior(behavior);
    }
    Ok(())
}
