use tracing::warn;

const FONT_NAME: &str = "vocotype-cjk";

pub(crate) fn install(ctx: &egui::Context) {
    let Some((path, bytes)) = load_font() else {
        warn!("没有找到可用中文字体, 悬浮窗可能显示方块");
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert(FONT_NAME.to_string(), egui::FontData::from_owned(bytes).into());
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, FONT_NAME.to_string());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, FONT_NAME.to_string());
    ctx.set_fonts(fonts);

    tracing::debug!(path, "悬浮窗中文字体已加载");
}

fn load_font() -> Option<(&'static str, Vec<u8>)> {
    font_candidates()
        .iter()
        .find_map(|path| std::fs::read(path).ok().map(|bytes| (*path, bytes)))
}

fn font_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    {
        &[
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ]
    }

    #[cfg(target_os = "windows")]
    {
        &[
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", unix)))]
    {
        &[]
    }
}
