use std::sync::Arc;

use crossbeam_channel::Receiver;
use tracing::{error, warn};

use crate::asr::{AsrEngine, TARGET_SAMPLE_RATE, TranscriptionResult};
use crate::dataset::DatasetRecorder;
use crate::inject::{InjectMethod, type_text};
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::SpeechSegment;

use super::state::{
    SharedRuntimeState, final_mode, finish_queue_item, overlay_state_with_lines,
};

pub(super) fn transcription_worker(
    engine: Arc<AsrEngine>,
    dataset: Option<DatasetRecorder>,
    overlay: OverlayHandle,
    state: SharedRuntimeState,
    segment_rx: Receiver<SpeechSegment>,
    inject_method: InjectMethod,
    append_newline: bool,
) {
    while let Ok(segment) = segment_rx.recv() {
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
                duration: pcm.duration_seconds(),
                inference_latency: 0.0,
                confidence: 0.0,
                error: Some(error.to_string()),
            },
        };

        if let Some(dataset) = &dataset
            && let Err(error) = dataset.record(&result, TARGET_SAMPLE_RATE, &segment.samples)
        {
            warn!(%error, "数据集记录失败");
        }

        let transcript = result.success.then_some(result.text.as_str());
        let (remaining, transcript_lines) = match finish_queue_item(&state, transcript) {
            Ok(result) => result,
            Err(error) => {
                error!(%error);
                continue;
            }
        };

        if result.success {
            if let Err(error) = type_text(&result.text, append_newline, inject_method.clone()) {
                overlay.set(overlay_state_with_lines(
                    OverlayMode::Error {
                        message: format!("文本注入失败: {}", error),
                    },
                    transcript_lines,
                ));
            } else {
                overlay.set(overlay_state_with_lines(
                    final_mode(remaining),
                    transcript_lines,
                ));
            }
        } else if result.is_empty_transcription() {
            overlay.set(overlay_state_with_lines(
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
            overlay.set(overlay_state_with_lines(
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
