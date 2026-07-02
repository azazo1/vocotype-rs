use super::ScreenRect;

pub(crate) fn current_mouse_screen_rect() -> Option<ScreenRect> {
    use x11rb::connection::Connection;
    use x11rb::protocol::randr::ConnectionExt as RandrConnectionExt;
    use x11rb::protocol::xproto::ConnectionExt as XprotoConnectionExt;

    let (connection, screen_index) = x11rb::connect(None).ok()?;
    let screen = &connection.setup().roots[screen_index];
    let pointer = connection.query_pointer(screen.root).ok()?.reply().ok()?;
    if !pointer.same_screen {
        return None;
    }

    let x = pointer.root_x as i32;
    let y = pointer.root_y as i32;
    if let Some(rect) = current_x11_monitor_rect(&connection, screen.root, x, y) {
        return Some(rect);
    }

    Some(ScreenRect {
        min: egui::pos2(0.0, 0.0),
        size: egui::vec2(
            screen.width_in_pixels as f32,
            screen.height_in_pixels as f32,
        ),
    })
}

fn current_x11_monitor_rect<C>(
    connection: &C,
    root: x11rb::protocol::xproto::Window,
    x: i32,
    y: i32,
) -> Option<ScreenRect>
where
    C: x11rb::connection::Connection
        + x11rb::protocol::randr::ConnectionExt,
{
    let monitors = connection
        .randr_get_monitors(root, true)
        .ok()?
        .reply()
        .ok()?
        .monitors;

    monitors.into_iter().find_map(|monitor| {
        let left = monitor.x as i32;
        let top = monitor.y as i32;
        let width = monitor.width as i32;
        let height = monitor.height as i32;
        let right = left + width;
        let bottom = top + height;
        if width <= 0
            || height <= 0
            || x < left
            || x >= right
            || y < top
            || y >= bottom
        {
            return None;
        }

        Some(ScreenRect {
            min: egui::pos2(left as f32, top as f32),
            size: egui::vec2(width as f32, height as f32),
        })
    })
}
