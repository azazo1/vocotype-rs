#[derive(Clone, Debug)]
pub enum OverlayMode {
    Idle,
    Recording { level: f32 },
    Silence { pending: usize },
    Transcribing { pending: usize },
    Streaming { revision: bool },
    Done,
    Error { message: String },
}

impl OverlayMode {
    pub(crate) fn title(&self) -> &'static str {
        match self {
            Self::Idle => "VocoType",
            Self::Recording { .. } => "正在录音",
            Self::Silence { .. } => "等待语音",
            Self::Transcribing { .. } => "正在转写",
            Self::Streaming { .. } => "实时转写",
            Self::Done => "转写完成",
            Self::Error { .. } => "需要处理",
        }
    }

    pub(crate) fn detail(&self) -> String {
        match self {
            Self::Idle => "按住热键开始本地转写".to_string(),
            Self::Recording { .. } => "继续说话, 停顿后会自动提交当前片段".to_string(),
            Self::Silence { pending } => format!("检测到停顿, 队列中有 {} 个片段", pending),
            Self::Transcribing { pending } => format!("后台转写中, 队列剩余 {}", pending),
            Self::Streaming { revision: true } => "已根据后文修正未落实内容".to_string(),
            Self::Streaming { revision: false } => "浅色内容仍可能被后文修正".to_string(),
            Self::Done => "本次转写完成".to_string(),
            Self::Error { message } => message.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StreamingTranscript {
    pub stable: String,
    pub unstable: String,
}

#[derive(Clone, Debug)]
pub struct OverlayState {
    pub mode: OverlayMode,
    pub transcript_lines: Vec<String>,
    pub streaming: Option<StreamingTranscript>,
}

impl OverlayState {
    pub fn new(mode: OverlayMode) -> Self {
        Self {
            mode,
            transcript_lines: Vec::new(),
            streaming: None,
        }
    }

    pub fn with_transcript(mode: OverlayMode, transcript_lines: Vec<String>) -> Self {
        Self {
            mode,
            transcript_lines,
            streaming: None,
        }
    }

    pub fn with_streaming(
        mode: OverlayMode,
        transcript_lines: Vec<String>,
        streaming: StreamingTranscript,
    ) -> Self {
        Self {
            mode,
            transcript_lines,
            streaming: Some(streaming),
        }
    }

    pub(crate) fn is_visible(&self) -> bool {
        if !self.transcript_lines.is_empty() || self.streaming.is_some() {
            return true;
        }
        !matches!(
            self.mode,
            OverlayMode::Idle | OverlayMode::Transcribing { pending: 0 }
        )
    }

    pub(crate) fn is_done(&self) -> bool {
        matches!(self.mode, OverlayMode::Done)
    }

    pub(crate) fn is_error(&self) -> bool {
        matches!(self.mode, OverlayMode::Error { .. })
    }
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            mode: OverlayMode::Idle,
            transcript_lines: Vec::new(),
            streaming: None,
        }
    }
}
