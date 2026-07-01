use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use enigo::{Enigo, KeyboardControllable};
use tracing::{debug, warn};

const CLIPBOARD_READY_DELAY: Duration = Duration::from_millis(80);
const CLIPBOARD_RESTORE_DELAY: Duration = Duration::from_millis(180);

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
        InjectMethod::Auto if should_prefer_clipboard(&payload) => clipboard_paste(&payload)
            .or_else(|error| {
                warn!(%error, "剪贴板粘贴失败, 尝试直接输入");
                direct_type(&payload)
            }),
        InjectMethod::Auto => direct_type(&payload).or_else(|error| {
            warn!(%error, "直接输入失败, 尝试剪贴板粘贴");
            clipboard_paste(&payload)
        }),
    }
}

fn should_prefer_clipboard(text: &str) -> bool {
    !text.is_ascii()
}

fn direct_type(text: &str) -> Result<()> {
    if should_prefer_clipboard(text) {
        bail!("直接输入不适合非 ASCII 文本");
    }

    run_enigo("直接输入", |enigo| {
        enigo.key_sequence(text);
    })
}

fn clipboard_paste(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new().map_err(|error| anyhow!(error))?;
    let old = clipboard.get_text().ok();
    clipboard
        .set_text(text.to_string())
        .map_err(|error| anyhow!(error))?;
    thread::sleep(CLIPBOARD_READY_DELAY);

    paste_shortcut()?;
    thread::sleep(CLIPBOARD_RESTORE_DELAY);

    if let Some(old) = old
        && let Err(error) = clipboard.set_text(old)
    {
        warn!(%error, "恢复剪贴板失败");
    }
    Ok(())
}

fn run_enigo(action: &str, f: impl FnOnce(&mut Enigo)) -> Result<()> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut enigo = Enigo::new();
        f(&mut enigo);
    }))
    .map_err(|payload| anyhow!("{}失败: {}", action, panic_message(payload)))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "未知 panic".to_string()
    }
}

#[cfg(target_os = "macos")]
fn paste_shortcut() -> Result<()> {
    let output = std::process::Command::new("osascript")
        .arg("-e")
        .arg(r#"tell application "System Events" to keystroke "v" using command down"#)
        .output()
        .map_err(|error| anyhow!("无法执行 osascript 粘贴: {}", error))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("osascript 粘贴失败: {}", stderr.trim());
    }
}

#[cfg(not(target_os = "macos"))]
fn paste_shortcut() -> Result<()> {
    use enigo::Key;

    run_enigo("剪贴板快捷键", |enigo| {
        enigo.key_down(Key::Control);
        enigo.key_click(Key::Layout('v'));
        enigo.key_up(Key::Control);
    })
}
