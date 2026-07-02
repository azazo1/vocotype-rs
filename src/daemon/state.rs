use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

pub(super) type SharedRuntimeState = Arc<Mutex<RuntimeState>>;

#[derive(Debug)]
pub(super) struct RuntimeState {
    recording: bool,
    queued: usize,
    active_text: String,
}

pub(super) fn new_state() -> SharedRuntimeState {
    Arc::new(Mutex::new(RuntimeState {
        recording: false,
        queued: 0,
        active_text: String::new(),
    }))
}

pub(super) fn increment_queue(state: &SharedRuntimeState) -> Result<usize> {
    let mut guard = state.lock().map_err(|_| anyhow!("状态锁已损坏"))?;
    guard.queued = guard.queued.saturating_add(1);
    Ok(guard.queued)
}

pub(super) fn finish_queue_item(
    state: &SharedRuntimeState,
    active_text: String,
) -> Result<usize> {
    let mut guard = state.lock().map_err(|_| anyhow!("状态锁已损坏"))?;
    guard.queued = guard.queued.saturating_sub(1);
    guard.active_text = active_text;
    Ok(guard.queued)
}

pub(super) fn pending_count(state: &SharedRuntimeState) -> usize {
    state.lock().map(|guard| guard.queued).unwrap_or(0)
}

pub(super) fn set_recording(state: &SharedRuntimeState, recording: bool) {
    if let Ok(mut guard) = state.lock() {
        guard.recording = recording;
    }
}
