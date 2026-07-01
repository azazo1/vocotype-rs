use anyhow::{Result, anyhow};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::{Code, HotKey},
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
        let code = parse_code(&self.key)?;
        Ok(HotKey::new(None, code))
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

fn parse_code(value: &str) -> Result<Code> {
    value
        .trim()
        .parse::<Code>()
        .map_err(|_| anyhow!("不支持的热键: {}", value))
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
