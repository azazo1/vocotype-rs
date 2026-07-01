use anyhow::{Result, anyhow};
use enigo::{Enigo, Key, KeyboardControllable};
use tracing::{debug, warn};

#[derive(Clone, Debug)]
pub enum InjectMethod {
    Auto,
    Type,
    Clipboard,
}

impl InjectMethod {
    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "type" => Self::Type,
            "clipboard" => Self::Clipboard,
            _ => Self::Auto,
        }
    }
}

pub fn type_text(text: &str, append_newline: bool, method: InjectMethod) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }

    let payload = if append_newline {
        format!("{}\n", text)
    } else {
        text.to_string()
    };
    debug!(chars = payload.chars().count(), "准备注入文本");

    match method {
        InjectMethod::Type => direct_type(&payload),
        InjectMethod::Clipboard => clipboard_paste(&payload),
        InjectMethod::Auto => direct_type(&payload).or_else(|error| {
            warn!(%error, "直接输入失败, 尝试剪贴板粘贴");
            clipboard_paste(&payload)
        }),
    }
}

fn direct_type(text: &str) -> Result<()> {
    let mut enigo = Enigo::new();
    enigo.key_sequence(text);
    Ok(())
}

fn clipboard_paste(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new().map_err(|error| anyhow!(error))?;
    let old = clipboard.get_text().ok();
    clipboard
        .set_text(text.to_string())
        .map_err(|error| anyhow!(error))?;

    let mut enigo = Enigo::new();
    paste_shortcut(&mut enigo);

    if let Some(old) = old
        && let Err(error) = clipboard.set_text(old)
    {
        warn!(%error, "恢复剪贴板失败");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn paste_shortcut(enigo: &mut Enigo) {
    enigo.key_down(Key::Meta);
    enigo.key_click(Key::Layout('v'));
    enigo.key_up(Key::Meta);
}

#[cfg(not(target_os = "macos"))]
fn paste_shortcut(enigo: &mut Enigo) {
    enigo.key_down(Key::Control);
    enigo.key_click(Key::Layout('v'));
    enigo.key_up(Key::Control);
}
