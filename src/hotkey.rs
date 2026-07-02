use anyhow::{Result, anyhow};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::HotKey,
};
use tracing::info;

#[derive(Clone, Debug)]
pub struct HotkeyConfig {
    pub key: String,
    pub end_key: Option<String>,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            key: "F2".to_string(),
            end_key: None,
        }
    }
}

impl HotkeyConfig {
    pub fn parse_hotkey(&self) -> Result<HotKey> {
        parse_hotkey(&self.key)
    }

    pub fn parse_end_hotkey(&self) -> Result<Option<HotKey>> {
        self.end_key.as_deref().map(parse_hotkey).transpose()
    }
}

pub struct HotkeyManager {
    _manager: GlobalHotKeyManager,
    _hotkey: HotKey,
    _end_hotkey: Option<HotKey>,
}

impl HotkeyManager {
    pub fn new(config: &HotkeyConfig) -> Result<Self> {
        let manager = GlobalHotKeyManager::new()?;
        let hotkey = config.parse_hotkey()?;
        let end_hotkey = config.parse_end_hotkey()?;
        if end_hotkey.is_some_and(|end_hotkey| end_hotkey.id() == hotkey.id()) {
            return Err(anyhow!("结束热键不能和触发热键相同"));
        }
        manager.register(hotkey)?;
        if let Some(end_hotkey) = end_hotkey {
            manager.register(end_hotkey)?;
            info!(
                hotkey = %hotkey.into_string(),
                end_hotkey = %end_hotkey.into_string(),
                "已注册全局热键"
            );
            return Ok(Self {
                _manager: manager,
                _hotkey: hotkey,
                _end_hotkey: Some(end_hotkey),
            });
        }

        info!(hotkey = %hotkey.into_string(), "已注册全局热键");
        Ok(Self {
            _manager: manager,
            _hotkey: hotkey,
            _end_hotkey: None,
        })
    }

    pub fn action(&self, event: global_hotkey::GlobalHotKeyEvent) -> Option<HotkeyEvent> {
        let role = if event.id() == self._hotkey.id() {
            HotkeyRole::Trigger
        } else if self
            ._end_hotkey
            .is_some_and(|hotkey| event.id() == hotkey.id())
        {
            HotkeyRole::End
        } else {
            return None;
        };
        Some(HotkeyEvent {
            role,
            action: recv_action(event),
        })
    }

    pub fn trigger_label(&self) -> String {
        self._hotkey.into_string()
    }

    pub fn end_label(&self) -> Option<String> {
        self._end_hotkey.map(HotKey::into_string)
    }

    pub fn events() -> &'static global_hotkey::GlobalHotKeyEventReceiver {
        GlobalHotKeyEvent::receiver()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HotkeyRole {
    Trigger,
    End,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HotkeyEvent {
    pub role: HotkeyRole,
    pub action: HotkeyAction,
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
