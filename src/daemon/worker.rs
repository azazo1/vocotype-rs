use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use tracing::{debug, error, info, warn};

use crate::asr::{
    AsrEngine, AsrOptions, TARGET_SAMPLE_RATE, TranscriptionResult, TranscriptionUpdate,
};
use crate::dataset::DatasetRecorder;
use crate::inject::{InjectMethod, type_text};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::SpeechSegment;

use super::state::{
    SharedRuntimeState, final_mode, finish_queue_item, overlay_state_with_lines,
    streaming_overlay_state,
};
use super::streaming::StablePrefixTracker;

pub(super) struct TranscriptionWorkerConfig {
    pub store: ModelStore,
    pub dataset: Option<DatasetRecorder>,
    pub overlay: OverlayHandle,
    pub state: SharedRuntimeState,
    pub inject_method: InjectMethod,
    pub append_newline: bool,
    pub idle_unload_secs: u64,
    pub asr_options: AsrOptions,
}

pub(super) enum TranscriptionTask {
    Segment(SpeechSegment),
    StreamStart(Vec<i16>),
    StreamAudio(Vec<i16>),
    StreamFinish(SpeechSegment),
}

struct LiveStreamInput<'a> {
    task_rx: &'a Receiver<TranscriptionTask>,
    initial: Option<Vec<i16>>,
    final_segment: Option<SpeechSegment>,
}

impl<'a> LiveStreamInput<'a> {
    fn new(task_rx: &'a Receiver<TranscriptionTask>, initial: Vec<i16>) -> Self {
        Self {
            task_rx,
            initial: Some(initial),
            final_segment: None,
        }
    }

    fn read(&mut self, buffer: &mut Vec<i16>) -> anyhow::Result<bool> {
        if let Some(initial) = self.initial.take() {
            buffer.extend(initial);
            return Ok(true);
        }
        match self.task_rx.recv() {
            Ok(TranscriptionTask::StreamAudio(samples)) => {
                buffer.extend(samples);
                Ok(true)
            }
            Ok(TranscriptionTask::StreamFinish(segment)) => {
                self.final_segment = Some(segment);
                Ok(false)
            }
            Ok(TranscriptionTask::Segment(_)) | Ok(TranscriptionTask::StreamStart(_)) => {
                anyhow::bail!("实时流式任务顺序无效")
            }
            Err(error) => Err(anyhow::anyhow!("实时流式输入通道已关闭: {}", error)),
        }
    }

    fn finish(mut self) -> Option<SpeechSegment> {
        self.final_segment
            .take()
            .or_else(|| drain_stream_finish(self.task_rx))
    }
}

struct StreamingOutput<'a> {
    config: &'a TranscriptionWorkerConfig,
    tracker: StablePrefixTracker,
    injection_error: Option<String>,
}

impl<'a> StreamingOutput<'a> {
    fn new(config: &'a TranscriptionWorkerConfig) -> Self {
        Self {
            config,
            tracker: StablePrefixTracker::default(),
            injection_error: None,
        }
    }

    fn handle(&mut self, update: TranscriptionUpdate) -> Result<(), anyhow::Error> {
        let presentation = self.tracker.update(&update);
        self.config.overlay.set(streaming_overlay_state(
            &self.config.state,
            presentation.stable,
            presentation.unstable,
            presentation.revision,
        ));
        if self.injection_error.is_none()
            && let Some(text) = final_injection_text(&update)
            && let Err(error) = type_text(
                text,
                self.config.append_newline,
                self.config.inject_method.clone(),
            )
        {
            self.injection_error = Some(error.to_string());
        }
        debug!(
            sequence = update.sequence,
            revision = update.revision,
            revision_count = update.revision_count,
            final_result = update.final_result,
            "流式转写结果已处理"
        );
        Ok(())
    }

    fn final_error(&self) -> Option<String> {
        self.injection_error.clone()
    }
}

fn final_injection_text(update: &TranscriptionUpdate) -> Option<&str> {
    (update.final_result && !update.result.text.is_empty()).then_some(update.result.text.as_str())
}

pub(super) fn transcription_worker(
    config: TranscriptionWorkerConfig,
    task_rx: Receiver<TranscriptionTask>,
) {
    let mut engine: Option<Arc<AsrEngine>> = None;
    let idle_timeout =
        (config.idle_unload_secs > 0).then(|| Duration::from_secs(config.idle_unload_secs));

    loop {
        let task = match recv_task(&task_rx, idle_timeout) {
            WorkerEvent::Task(task) => task,
            WorkerEvent::Idle => {
                if engine.take().is_some() {
                    info!(idle_secs = config.idle_unload_secs, "空闲卸载 ASR 和 PUNC 模型");
                }
                continue;
            }
            WorkerEvent::Closed => break,
        };
        let engine = match ensure_engine(&mut engine, &config.store, config.asr_options.clone()) {
            Ok(engine) => engine,
            Err(error) => {
                if matches!(&task, TranscriptionTask::StreamStart(_)) {
                    let _ = drain_stream_finish(&task_rx);
                }
                handle_worker_error(
                    &config.overlay,
                    &config.state,
                    format!("ASR 模型加载失败: {}", error),
                );
                continue;
            }
        };
        let (segment, transcribed, streaming, stream_injection_error) = match task {
            TranscriptionTask::Segment(segment) => {
                let pcm = crate::wav::PcmAudio {
                    sample_rate: TARGET_SAMPLE_RATE,
                    samples: segment.samples.clone(),
                };
                if engine.supports_streaming() {
                    let mut output = StreamingOutput::new(&config);
                    let transcribed = engine
                        .transcribe_pcm_streaming(&pcm, |update| output.handle(update));
                    (segment, transcribed, true, output.final_error())
                } else {
                    let transcribed = engine.transcribe_pcm(&pcm);
                    (segment, transcribed, false, None)
                }
            }
            TranscriptionTask::StreamStart(initial) => {
                if !engine.supports_streaming() {
                    let _ = drain_stream_finish(&task_rx);
                    handle_worker_error(
                        &config.overlay,
                        &config.state,
                        "当前 ASR 后端不支持实时流式输入".to_string(),
                    );
                    continue;
                }
                let mut input = LiveStreamInput::new(&task_rx, initial);
                let mut output = StreamingOutput::new(&config);
                let transcribed = engine.transcribe_live_pcm(
                    |buffer| input.read(buffer),
                    |update| output.handle(update),
                );
                let Some(segment) = input.finish() else {
                    handle_worker_error(
                        &config.overlay,
                        &config.state,
                        "实时流式输入没有结束语音段".to_string(),
                    );
                    continue;
                };
                (segment, transcribed, true, output.final_error())
            }
            TranscriptionTask::StreamAudio(_) | TranscriptionTask::StreamFinish(_) => {
                warn!("收到没有开始事件的实时流式任务");
                continue;
            }
        };
        let result = match transcribed {
            Ok(result) => result,
            Err(error) => TranscriptionResult {
                success: false,
                text: String::new(),
                raw_text: String::new(),
                tokens: Vec::new(),
                token_timestamps: None,
                duration: segment.speech_ms as f32 / 1_000.0,
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

        let output_text = result.text.clone();
        let transcript = result.success.then_some(output_text.as_str());
        let (remaining, transcript_lines) = match finish_queue_item(&config.state, transcript) {
            Ok(result) => result,
            Err(error) => {
                error!(%error);
                continue;
            }
        };

        if result.success {
            let injection_error = if streaming {
                stream_injection_error
            } else {
                type_text(
                    &output_text,
                    config.append_newline,
                    config.inject_method.clone(),
                )
                .err()
                .map(|error| error.to_string())
            };
            if let Some(error) = injection_error {
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
    Task(TranscriptionTask),
    Idle,
    Closed,
}

fn recv_task(
    task_rx: &Receiver<TranscriptionTask>,
    idle_timeout: Option<Duration>,
) -> WorkerEvent {
    match idle_timeout {
        Some(timeout) => match task_rx.recv_timeout(timeout) {
            Ok(task) => WorkerEvent::Task(task),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => WorkerEvent::Idle,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => WorkerEvent::Closed,
        },
        None => match task_rx.recv() {
            Ok(task) => WorkerEvent::Task(task),
            Err(_) => WorkerEvent::Closed,
        },
    }
}

fn drain_stream_finish(task_rx: &Receiver<TranscriptionTask>) -> Option<SpeechSegment> {
    loop {
        match task_rx.recv().ok()? {
            TranscriptionTask::StreamFinish(segment) => return Some(segment),
            TranscriptionTask::StreamAudio(_) => {}
            TranscriptionTask::Segment(_) | TranscriptionTask::StreamStart(_) => return None,
        }
    }
}

fn ensure_engine(
    engine: &mut Option<Arc<AsrEngine>>,
    store: &ModelStore,
    asr_options: AsrOptions,
) -> anyhow::Result<Arc<AsrEngine>> {
    if let Some(engine) = engine {
        return Ok(engine.clone());
    }
    debug!("懒加载 ASR 和 PUNC 模型");
    let loaded = AsrEngine::load_with_options(store.clone(), asr_options)?;
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
    use crate::asr::TranscriptionResult;
    use crate::vad::SegmentReason;

    fn segment(sample: i16) -> SpeechSegment {
        SpeechSegment {
            samples: vec![sample],
            reason: SegmentReason::Finish,
            speech_ms: 1,
            start_sample: 0,
            end_sample: 1,
            audio_start_sample: 0,
            audio_end_sample: 1,
        }
    }

    fn transcription_update(text: &str, final_result: bool) -> TranscriptionUpdate {
        TranscriptionUpdate {
            result: TranscriptionResult {
                success: !text.is_empty(),
                text: text.to_string(),
                raw_text: text.to_string(),
                tokens: Vec::new(),
                token_timestamps: None,
                duration: 1.0,
                inference_latency: 0.1,
                confidence: 1.0,
                error: None,
            },
            revision: false,
            revision_count: 0,
            sequence: 1,
            final_result,
        }
    }

    #[test]
    fn recv_task_reports_idle_when_timeout_expires() {
        let (_tx, rx) = crossbeam_channel::unbounded();
        let event = recv_task(&rx, Some(Duration::from_millis(1)));
        assert!(matches!(event, WorkerEvent::Idle));
    }

    #[test]
    fn recv_task_reports_closed_without_idle_timeout() {
        let (tx, rx) = crossbeam_channel::unbounded();
        drop(tx);
        let event = recv_task(&rx, None);
        assert!(matches!(event, WorkerEvent::Closed));
    }

    #[test]
    fn live_stream_input_keeps_consecutive_streams_separate() {
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(TranscriptionTask::StreamAudio(vec![2])).unwrap();
        tx.send(TranscriptionTask::StreamFinish(segment(3))).unwrap();
        tx.send(TranscriptionTask::StreamStart(vec![4])).unwrap();

        let mut input = LiveStreamInput::new(&rx, vec![1]);
        let mut buffer = Vec::new();
        assert!(input.read(&mut buffer).unwrap());
        assert!(input.read(&mut buffer).unwrap());
        assert!(!input.read(&mut buffer).unwrap());
        assert_eq!(buffer, vec![1, 2]);
        assert_eq!(input.finish().unwrap().samples, vec![3]);
        assert!(matches!(
            rx.recv().unwrap(),
            TranscriptionTask::StreamStart(samples) if samples == vec![4]
        ));
    }

    #[test]
    fn drain_stream_finish_stops_before_the_next_task() {
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(TranscriptionTask::StreamAudio(vec![1])).unwrap();
        tx.send(TranscriptionTask::StreamFinish(segment(2))).unwrap();
        tx.send(TranscriptionTask::Segment(segment(3))).unwrap();

        assert_eq!(drain_stream_finish(&rx).unwrap().samples, vec![2]);
        assert!(matches!(
            rx.recv().unwrap(),
            TranscriptionTask::Segment(segment) if segment.samples == vec![3]
        ));
    }

    #[test]
    fn live_stream_input_reports_a_closed_channel() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut input = LiveStreamInput::new(&rx, vec![1]);
        let mut buffer = Vec::new();
        assert!(input.read(&mut buffer).unwrap());
        drop(tx);
        assert!(input.read(&mut buffer).is_err());
    }

    #[test]
    fn injects_only_the_complete_final_text() {
        let partial = transcription_update("实时结果", false);
        assert!(final_injection_text(&partial).is_none());

        let final_result = transcription_update("最终完整结果", true);
        assert_eq!(final_injection_text(&final_result), Some("最终完整结果"));

        let empty_final = transcription_update("", true);
        assert!(final_injection_text(&empty_final).is_none());
    }

}
