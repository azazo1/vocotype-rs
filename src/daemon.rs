use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use tracing::{error, info, warn};

use crate::asr::{AsrEngine, TARGET_SAMPLE_RATE, TranscriptionResult};
use crate::audio::AudioInput;
use crate::dataset::DatasetRecorder;
use crate::hotkey::{HotkeyAction, HotkeyConfig, HotkeyManager, recv_action};
use crate::inject::{InjectMethod, type_text};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode, OverlayState, create as create_overlay};
use crate::vad::{SpeechSegment, VadConfig, VadSegmenter};

#[derive(Clone, Debug)]
pub struct DaemonOptions {
    pub hotkey: String,
    pub save_dataset: bool,
    pub dataset_dir: Option<PathBuf>,
    pub append_newline: bool,
    pub inject_method: InjectMethod,
    pub end_silence_ms: u32,
    pub pre_roll_ms: u32,
    pub tail_padding_ms: u32,
    pub min_speech_ms: u32,
    pub max_segment_ms: u32,
}

#[derive(Debug)]
struct RuntimeState {
    recording: bool,
    queued: usize,
    active_text: String,
}

pub async fn run_daemon(store: ModelStore, options: DaemonOptions) -> Result<()> {
    store.paths.ensure_dirs()?;
    if let Err(error) = store.verify_required() {
        error!(%error, "模型缺失");
        eprintln!("模型缺失, 请先运行: {}", store.download_hint());
        return Err(error);
    }

    let (overlay, overlay_runner) = create_overlay();
    overlay.idle();
    let daemon_overlay = overlay.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(error) = run_daemon_loop(store, options, daemon_overlay) {
            error!(%error, "daemon 运行失败");
            overlay.set(OverlayState {
                mode: OverlayMode::Error {
                    message: error.to_string(),
                },
            });
        }
    });

    overlay_runner.run().context("无法启动悬浮窗")
}

fn run_daemon_loop(
    store: ModelStore,
    options: DaemonOptions,
    overlay: OverlayHandle,
) -> Result<()> {
    let engine = AsrEngine::load(store.clone())?;
    let mut segmenter = build_segmenter(&options, &store)?;
    let dataset = if options.save_dataset {
        let dir = options
            .dataset_dir
            .clone()
            .unwrap_or_else(|| store.paths.data_dir.join("dataset"));
        Some(DatasetRecorder::new(dir)?)
    } else {
        None
    };

    let hotkey_cfg = HotkeyConfig {
        key: options.hotkey.clone(),
    };
    let _hotkeys = HotkeyManager::new(&hotkey_cfg)?;
    let state = Arc::new(Mutex::new(RuntimeState {
        recording: false,
        queued: 0,
        active_text: String::new(),
    }));

    let (segment_tx, segment_rx) = unbounded::<SpeechSegment>();
    let overlay_worker = overlay.clone();
    let state_worker = state.clone();
    let inject_method = options.inject_method.clone();
    let append_newline = options.append_newline;
    tokio::task::spawn_blocking(move || {
        transcription_worker(
            engine,
            dataset,
            overlay_worker,
            state_worker,
            segment_rx,
            inject_method,
            append_newline,
        );
    });

    let hotkey_events = HotkeyManager::events().clone();
    let mut capturing = false;
    let mut audio_input: Option<AudioInput> = None;
    let mut audio_rx: Option<Receiver<Vec<i16>>> = None;
    let mut last_frame_at = Instant::now();

    info!("daemon 已启动, 按住热键开始录音");

    loop {
        crossbeam_channel::select! {
            recv(hotkey_events) -> event => {
                let event = event.context("热键事件通道已关闭")?;
                match recv_action(event) {
                    HotkeyAction::Pressed if !capturing => {
                        info!("热键按下, 开始录音");
                        let input = AudioInput::start(None)?;
                        audio_rx = Some(input.receiver());
                        audio_input = Some(input);
                        segmenter.reset();
                        capturing = true;
                        last_frame_at = Instant::now();
                        set_recording(&state, true);
                        overlay.set(OverlayState { mode: OverlayMode::Recording { level: 0.0 } });
                    }
                    HotkeyAction::Released if capturing => {
                        info!("热键释放, 结束录音");
                        capturing = false;
                        set_recording(&state, false);
                        for segment in segmenter.finish() {
                            submit_segment(&segment_tx, &state, &overlay, segment)?;
                        }
                        if let Some(input) = audio_input.take() {
                            input.stop();
                        }
                        audio_rx = None;
                        overlay.set(OverlayState { mode: OverlayMode::Transcribing { pending: pending_count(&state) } });
                    }
                    _ => {}
                }
            }
            default(Duration::from_millis(30)) => {
                if capturing && let Some(rx) = &audio_rx {
                    match rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(frame) => {
                            last_frame_at = Instant::now();
                            let level = frame_level(&frame);
                            let mode = if segmenter.detected() {
                                OverlayMode::Recording { level }
                            } else {
                                OverlayMode::Silence { pending: pending_count(&state) }
                            };
                            overlay.set(OverlayState { mode });
                            for segment in segmenter.push(&frame) {
                                submit_segment(&segment_tx, &state, &overlay, segment)?;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if last_frame_at.elapsed() > Duration::from_millis(options.end_silence_ms as u64) {
                                overlay.set(OverlayState { mode: OverlayMode::Silence { pending: pending_count(&state) } });
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            warn!("音频流断开");
                            capturing = false;
                            set_recording(&state, false);
                            if let Some(input) = audio_input.take() {
                                input.stop();
                            }
                            audio_rx = None;
                        }
                    }
                }
            }
        }
    }
}

fn build_segmenter(options: &DaemonOptions, store: &ModelStore) -> Result<VadSegmenter> {
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

fn submit_segment(
    segment_tx: &Sender<SpeechSegment>,
    state: &Arc<Mutex<RuntimeState>>,
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

    let pending = {
        let mut guard = state.lock().map_err(|_| anyhow!("状态锁已损坏"))?;
        guard.queued = guard.queued.saturating_add(1);
        guard.queued
    };
    overlay.set(OverlayState {
        mode: OverlayMode::Transcribing { pending },
    });
    segment_tx
        .send(segment)
        .map_err(|error| anyhow!("无法提交转写任务: {}", error))?;
    Ok(())
}

fn transcription_worker(
    engine: Arc<AsrEngine>,
    dataset: Option<DatasetRecorder>,
    overlay: OverlayHandle,
    state: Arc<Mutex<RuntimeState>>,
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

        {
            let mut guard = match state.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    error!("状态锁已损坏");
                    continue;
                }
            };
            guard.queued = guard.queued.saturating_sub(1);
            guard.active_text = result.text.clone();
        }

        if result.success {
            overlay.set(OverlayState {
                mode: OverlayMode::Done {
                    text: result.text.clone(),
                },
            });
            if let Err(error) = type_text(&result.text, append_newline, inject_method.clone()) {
                overlay.set(OverlayState {
                    mode: OverlayMode::Error {
                        message: format!("文本注入失败: {}", error),
                    },
                });
            }
        } else {
            overlay.set(OverlayState {
                mode: OverlayMode::Error {
                    message: result
                        .error
                        .clone()
                        .unwrap_or_else(|| "转写失败".to_string()),
                },
            });
        }
    }
}

fn frame_level(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let peak = frame.iter().map(|sample| sample.abs()).max().unwrap_or(0) as f32;
    (peak / i16::MAX as f32).clamp(0.0, 1.0)
}

fn pending_count(state: &Arc<Mutex<RuntimeState>>) -> usize {
    state.lock().map(|guard| guard.queued).unwrap_or(0)
}

fn set_recording(state: &Arc<Mutex<RuntimeState>>, recording: bool) {
    if let Ok(mut guard) = state.lock() {
        guard.recording = recording;
    }
}
