use anyhow::{Result, anyhow};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::HotKey,
};
use tracing::info;

#[derive(Clone, Debug)]
pub struct HotkeyConfig {
    pub key: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            key: "F2".to_string(),
        }
    }
}

impl HotkeyConfig {
    pub fn parse_hotkey(&self) -> Result<HotKey> {
        parse_hotkey(&self.key)
    }
}

pub struct HotkeyManager {
    _manager: GlobalHotKeyManager,
    _hotkey: HotKey,
}

impl HotkeyManager {
    pub fn new(config: &HotkeyConfig) -> Result<Self> {
        let manager = GlobalHotKeyManager::new()?;
        let hotkey = config.parse_hotkey()?;
        manager.register(hotkey)?;
        info!(hotkey = %hotkey.into_string(), "已注册全局热键");
        Ok(Self {
            _manager: manager,
            _hotkey: hotkey,
        })
    }

    pub fn events() -> &'static global_hotkey::GlobalHotKeyEventReceiver {
        GlobalHotKeyEvent::receiver()
    }
}

fn parse_hotkey(value: &str) -> Result<HotKey> {
    let normalized = normalize_hotkey(value);
    normalized
        .parse::<HotKey>()
        .map_err(|error| anyhow!("不支持的热键组合: {}. {}", value, error))
}

fn normalize_hotkey(value: &str) -> String {
    value
        .split('+')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("+")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HotkeyAction {
    Pressed,
    Released,
}

pub fn recv_action(event: global_hotkey::GlobalHotKeyEvent) -> HotkeyAction {
    match event.state() {
        global_hotkey::HotKeyState::Pressed => HotkeyAction::Pressed,
        global_hotkey::HotKeyState::Released => HotkeyAction::Released,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use global_hotkey::hotkey::{Code, Modifiers};

    #[test]
    fn parses_single_key_hotkey() {
        let hotkey = parse_hotkey("F2").unwrap();

        assert_eq!(hotkey.mods, Modifiers::empty());
        assert_eq!(hotkey.key, Code::F2);
    }

    #[test]
    fn parses_modifier_combo_hotkey() {
        let hotkey = parse_hotkey("ctrl + f2").unwrap();

        assert_eq!(hotkey.mods, Modifiers::CONTROL);
        assert_eq!(hotkey.key, Code::F2);
    }

    #[test]
    fn parses_multiple_modifiers_hotkey() {
        let hotkey = parse_hotkey("shift+alt+space").unwrap();

        assert_eq!(hotkey.mods, Modifiers::SHIFT | Modifiers::ALT);
        assert_eq!(hotkey.key, Code::Space);
    }

    #[test]
    fn rejects_modifier_after_key() {
        assert!(parse_hotkey("f2+ctrl").is_err());
    }
}
