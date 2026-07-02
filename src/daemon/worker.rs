use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use tracing::{debug, error, info, warn};

use crate::asr::{AsrEngine, TARGET_SAMPLE_RATE, TranscriptionResult};
use crate::dataset::DatasetRecorder;
use crate::inject::{InjectMethod, type_text};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::SpeechSegment;

use super::state::{
    SharedRuntimeState, final_mode, finish_queue_item, overlay_state_with_lines,
};

pub(super) struct TranscriptionWorkerConfig {
    pub store: ModelStore,
    pub dataset: Option<DatasetRecorder>,
    pub overlay: OverlayHandle,
    pub state: SharedRuntimeState,
    pub inject_method: InjectMethod,
    pub append_newline: bool,
    pub idle_unload_secs: u64,
}

pub(super) fn transcription_worker(
    config: TranscriptionWorkerConfig,
    segment_rx: Receiver<SpeechSegment>,
) {
    let mut engine: Option<Arc<AsrEngine>> = None;
    let idle_timeout =
        (config.idle_unload_secs > 0).then(|| Duration::from_secs(config.idle_unload_secs));

    loop {
        let segment = match recv_segment(&segment_rx, idle_timeout) {
            WorkerEvent::Segment(segment) => segment,
            WorkerEvent::Idle => {
                if engine.take().is_some() {
                    info!(idle_secs = config.idle_unload_secs, "空闲卸载 ASR 和 PUNC 模型");
                }
                continue;
            }
            WorkerEvent::Closed => break,
        };
        let engine = match ensure_engine(&mut engine, &config.store) {
            Ok(engine) => engine,
            Err(error) => {
                handle_worker_error(
                    &config.overlay,
                    &config.state,
                    format!("ASR 模型加载失败: {}", error),
                );
                continue;
            }
        };
        let pcm = crate::wav::PcmAudio {
            sample_rate: TARGET_SAMPLE_RATE,
            samples: segment.samples.clone(),
        };
        let result = match engine.transcribe_pcm(&pcm) {
            Ok(result) => result,
            Err(error) => TranscriptionResult {
                success: false,
                text: String::new(),
                raw_text: String::new(),
                tokens: Vec::new(),
                token_timestamps: None,
                duration: pcm.duration_seconds(),
                inference_latency: 0.0,
                confidence: 0.0,
                error: Some(error.to_string()),
            },
        };

        if let Some(dataset) = &config.dataset
            && let Err(error) = dataset.record(&result, TARGET_SAMPLE_RATE, &segment.samples)
        {
            warn!(%error, "数据集记录失败");
        }

        let transcript = result.success.then_some(result.text.as_str());
        let (remaining, transcript_lines) = match finish_queue_item(&config.state, transcript) {
            Ok(result) => result,
            Err(error) => {
                error!(%error);
                continue;
            }
        };

        if result.success {
            if let Err(error) =
                type_text(&result.text, config.append_newline, config.inject_method.clone())
            {
                config.overlay.set(overlay_state_with_lines(
                    OverlayMode::Error {
                        message: format!("文本注入失败: {}", error),
                    },
                    transcript_lines,
                ));
            } else {
                config.overlay.set(overlay_state_with_lines(
                    final_mode(remaining),
                    transcript_lines,
                ));
            }
        } else if result.is_empty_transcription() {
            config.overlay.set(overlay_state_with_lines(
                final_mode(remaining),
                transcript_lines,
            ));
        } else {
            let duration_label = format!("{:.2}", result.duration);
            warn!(
                error = result.error.as_deref().unwrap_or("unknown"),
                duration = %duration_label,
                "语音段转写失败"
            );
            config.overlay.set(overlay_state_with_lines(
                OverlayMode::Error {
                    message: result
                        .error
                        .clone()
                        .unwrap_or_else(|| "转写失败".to_string()),
                },
                transcript_lines,
            ));
        }
    }
}

enum WorkerEvent {
    Segment(SpeechSegment),
    Idle,
    Closed,
}

fn recv_segment(
    segment_rx: &Receiver<SpeechSegment>,
    idle_timeout: Option<Duration>,
) -> WorkerEvent {
    match idle_timeout {
        Some(timeout) => match segment_rx.recv_timeout(timeout) {
            Ok(segment) => WorkerEvent::Segment(segment),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => WorkerEvent::Idle,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => WorkerEvent::Closed,
        },
        None => match segment_rx.recv() {
            Ok(segment) => WorkerEvent::Segment(segment),
            Err(_) => WorkerEvent::Closed,
        },
    }
}

fn ensure_engine(
    engine: &mut Option<Arc<AsrEngine>>,
    store: &ModelStore,
) -> anyhow::Result<Arc<AsrEngine>> {
    if let Some(engine) = engine {
        return Ok(engine.clone());
    }
    debug!("懒加载 ASR 和 PUNC 模型");
    let loaded = AsrEngine::load(store.clone())?;
    *engine = Some(loaded.clone());
    Ok(loaded)
}

fn handle_worker_error(
    overlay: &OverlayHandle,
    state: &SharedRuntimeState,
    message: String,
) {
    let (remaining, transcript_lines) = match finish_queue_item(state, None) {
        Ok(result) => result,
        Err(error) => {
            error!(%error);
            return;
        }
    };
    let mode = if remaining == 0 {
        OverlayMode::Error { message }
    } else {
        final_mode(remaining)
    };
    overlay.set(overlay_state_with_lines(mode, transcript_lines));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recv_segment_reports_idle_when_timeout_expires() {
        let (_tx, rx) = crossbeam_channel::unbounded();
        let event = recv_segment(&rx, Some(Duration::from_millis(1)));
        assert!(matches!(event, WorkerEvent::Idle));
    }

    #[test]
    fn recv_segment_reports_closed_without_idle_timeout() {
        let (tx, rx) = crossbeam_channel::unbounded();
        drop(tx);
        let event = recv_segment(&rx, None);
        assert!(matches!(event, WorkerEvent::Closed));
    }
}
