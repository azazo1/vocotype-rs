use anyhow::{Result, anyhow};
use crossbeam_channel::Sender;
use tracing::info;

use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::{SpeechSegment, VadConfig, VadSegmenter};

use super::DaemonOptions;
use super::state::{SharedRuntimeState, increment_queue, overlay_state};
use super::worker::TranscriptionTask;

pub(super) fn build_segmenter(
    options: &DaemonOptions,
    store: &ModelStore,
) -> Result<VadSegmenter> {
    if options.asr_options.backend == crate::asr_backend::AsrBackend::Sherpa {
        store.verify_vad_checksum()?;
    }
    let model_path = store.vad_model_path_for(options.asr_options.backend)?;
    VadSegmenter::new_for_backend(
        options.asr_options.backend,
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
    task_tx: &Sender<TranscriptionTask>,
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
    overlay.set(overlay_state(state, OverlayMode::Transcribing { pending }));
    task_tx
        .send(TranscriptionTask::Segment(segment))
        .map_err(|error| anyhow!("无法提交转写任务: {}", error))?;
    Ok(())
}

pub(super) fn submit_stream_start(
    task_tx: &Sender<TranscriptionTask>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    samples: Vec<i16>,
) -> Result<bool> {
    if samples.is_empty() {
        return Ok(false);
    }
    let pending = increment_queue(state)?;
    overlay.set(overlay_state(state, OverlayMode::Transcribing { pending }));
    task_tx
        .send(TranscriptionTask::StreamStart(samples))
        .map_err(|error| anyhow!("无法开始实时流式转写: {}", error))?;
    Ok(true)
}

pub(super) fn submit_stream_audio(
    task_tx: &Sender<TranscriptionTask>,
    samples: Vec<i16>,
) -> Result<()> {
    task_tx
        .send(TranscriptionTask::StreamAudio(samples))
        .map_err(|error| anyhow!("无法提交实时流式音频: {}", error))
}

pub(super) fn submit_stream_finish(
    task_tx: &Sender<TranscriptionTask>,
    segment: SpeechSegment,
) -> Result<()> {
    task_tx
        .send(TranscriptionTask::StreamFinish(segment))
        .map_err(|error| anyhow!("无法结束实时流式转写: {}", error))
}

pub(super) fn frame_level(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let peak = frame.iter().map(|sample| sample.abs()).max().unwrap_or(0) as f32;
    (peak / i16::MAX as f32).clamp(0.0, 1.0)
}
