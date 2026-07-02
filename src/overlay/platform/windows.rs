use std::mem::size_of;

use super::ScreenRect;

pub(crate) fn current_mouse_screen_rect() -> Option<ScreenRect> {
    use windows_sys::Win32::Foundation::{POINT, RECT};
    use windows_sys::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromPoint,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;

    unsafe {
        let mut point = POINT::default();
        if GetCursorPos(&mut point) == 0 {
            return None;
        }

        let monitor = MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST);
        if monitor.is_null() {
            return None;
        }

        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(monitor, &mut info) == 0 {
            return None;
        }

        screen_rect_from_work_area(info.rcWork)
    }
}

fn screen_rect_from_work_area(rect: RECT) -> Option<ScreenRect> {
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 {
        return None;
    }

    Some(ScreenRect {
        min: egui::pos2(rect.left as f32, rect.top as f32),
        size: egui::vec2(width as f32, height as f32),
    })
}
