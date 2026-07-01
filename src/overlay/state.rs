#[derive(Clone, Debug)]
pub enum OverlayMode {
    Idle,
    Recording { level: f32 },
    Silence { pending: usize },
    Transcribing { pending: usize },
    Done { text: String },
    Error { message: String },
}

impl OverlayMode {
    pub(crate) fn is_visible(&self) -> bool {
        !matches!(
            self,
            Self::Idle | Self::Transcribing { pending: 0 }
        )
    }

    pub(crate) fn is_done(&self) -> bool {
        matches!(self, Self::Done { .. })
    }

    pub(crate) fn title(&self) -> &'static str {
        match self {
            Self::Idle => "VocoType",
            Self::Recording { .. } => "正在录音",
            Self::Silence { .. } => "等待语音",
            Self::Transcribing { .. } => "正在转写",
            Self::Done { .. } => "转写完成",
            Self::Error { .. } => "需要处理",
        }
    }

    pub(crate) fn detail(&self) -> String {
        match self {
            Self::Idle => "按住热键开始本地转写".to_string(),
            Self::Recording { .. } => "继续说话, 停顿后会自动提交当前片段".to_string(),
            Self::Silence { pending } => format!("检测到停顿, 队列中有 {} 个片段", pending),
            Self::Transcribing { pending } => format!("后台转写中, 队列剩余 {}", pending),
            Self::Done { text } => preview_text(text),
            Self::Error { message } => message.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OverlayState {
    pub mode: OverlayMode,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            mode: OverlayMode::Idle,
        }
    }
}

fn preview_text(text: &str) -> String {
    const MAX_CHARS: usize = 36;

    let mut chars = text.chars();
    let preview = chars.by_ref().take(MAX_CHARS).collect::<String>();
    if chars.next().is_some() {
        format!("{}...", preview)
    } else {
        preview
    }
}
