use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender};
use tracing::info;

use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode, OverlayState};
use crate::vad::{SpeechSegment, VadConfig, VadSegmenter};

use super::DaemonOptions;
use super::state::{SharedRuntimeState, increment_queue};

pub(super) fn build_segmenter(
    options: &DaemonOptions,
    store: &ModelStore,
) -> Result<VadSegmenter> {
    store.verify_vad_checksum()?;
    let model_path = store.vad_model_path()?;
    VadSegmenter::new(
        VadConfig {
            end_silence_ms: options.end_silence_ms,
            pre_roll_ms: options.pre_roll_ms,
            tail_padding_ms: options.tail_padding_ms,
            min_speech_ms: options.min_speech_ms,
            max_segment_ms: options.max_segment_ms,
            ..VadConfig::default()
        },
        &model_path,
    )
}

pub(super) fn submit_segment(
    segment_tx: &Sender<SpeechSegment>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    segment: SpeechSegment,
) -> Result<()> {
    if segment.samples.is_empty() {
        return Ok(());
    }
    info!(
        reason = ?segment.reason,
        speech_ms = segment.speech_ms,
        samples = segment.samples.len(),
        "提交语音段转写"
    );

    let pending = increment_queue(state)?;
    overlay.set(OverlayState {
        mode: OverlayMode::Transcribing { pending },
    });
    segment_tx
        .send(segment)
        .map_err(|error| anyhow!("无法提交转写任务: {}", error))?;
    Ok(())
}

pub(super) fn drain_release_tail(
    audio_rx: &Receiver<Vec<i16>>,
    segmenter: &mut VadSegmenter,
    segment_tx: &Sender<SpeechSegment>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    wait: Duration,
) -> Result<()> {
    let deadline = Instant::now() + wait;
    while Instant::now() < deadline {
        let timeout = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(10));
        match audio_rx.recv_timeout(timeout) {
            Ok(frame) => {
                for segment in segmenter.push(&frame) {
                    submit_segment(segment_tx, state, overlay, segment)?;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

pub(super) fn frame_level(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let peak = frame.iter().map(|sample| sample.abs()).max().unwrap_or(0) as f32;
    (peak / i16::MAX as f32).clamp(0.0, 1.0)
}
