use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::overlay::{OverlayMode, OverlayState};

pub(super) type SharedRuntimeState = Arc<Mutex<RuntimeState>>;

#[derive(Debug)]
pub(super) struct RuntimeState {
    recording: bool,
    queued: usize,
    transcript_lines: Vec<String>,
}

pub(super) fn new_state() -> SharedRuntimeState {
    Arc::new(Mutex::new(RuntimeState {
        recording: false,
        queued: 0,
        transcript_lines: Vec::new(),
    }))
}

pub(super) fn begin_recording_session(state: &SharedRuntimeState) {
    if let Ok(mut guard) = state.lock() {
        guard.recording = true;
        guard.transcript_lines.clear();
    }
}

pub(super) fn end_recording_session(state: &SharedRuntimeState) {
    if let Ok(mut guard) = state.lock() {
        guard.recording = false;
        guard.transcript_lines.clear();
    }
}

pub(super) fn increment_queue(state: &SharedRuntimeState) -> Result<usize> {
    let mut guard = state.lock().map_err(|_| anyhow!("状态锁已损坏"))?;
    guard.queued = guard.queued.saturating_add(1);
    Ok(guard.queued)
}

pub(super) fn finish_queue_item(
    state: &SharedRuntimeState,
    transcript: Option<&str>,
) -> Result<(usize, Vec<String>)> {
    let mut guard = state.lock().map_err(|_| anyhow!("状态锁已损坏"))?;
    guard.queued = guard.queued.saturating_sub(1);
    if guard.recording
        && let Some(transcript) = transcript
        && !transcript.trim().is_empty()
    {
        guard.transcript_lines.push(transcript.to_string());
    }
    Ok((guard.queued, guard.transcript_lines.clone()))
}

pub(super) fn pending_count(state: &SharedRuntimeState) -> usize {
    state.lock().map(|guard| guard.queued).unwrap_or(0)
}

pub(super) fn overlay_state(state: &SharedRuntimeState, mode: OverlayMode) -> OverlayState {
    let transcript_lines = state
        .lock()
        .map(|guard| guard.transcript_lines.clone())
        .unwrap_or_default();
    OverlayState::with_transcript(mode, transcript_lines)
}

pub(super) fn overlay_state_with_lines(
    mode: OverlayMode,
    transcript_lines: Vec<String>,
) -> OverlayState {
    OverlayState::with_transcript(mode, transcript_lines)
}

pub(super) fn final_mode(remaining: usize) -> OverlayMode {
    if remaining == 0 {
        OverlayMode::Done
    } else {
        OverlayMode::Transcribing { pending: remaining }
    }
}
