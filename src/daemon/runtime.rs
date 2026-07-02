use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender};
use tracing::{debug, info, warn};

use crate::audio::AudioInput;
use crate::dataset::DatasetRecorder;
use crate::hotkey::{HotkeyAction, HotkeyConfig, HotkeyEvent, HotkeyManager, HotkeyRole};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::{SpeechSegment, VadSegmenter};

use super::{DaemonOptions, HotkeyMode};
use super::segments::{
    build_segmenter, drain_release_tail, frame_level, submit_segment,
};
use super::state::{
    SharedRuntimeState, begin_recording_session, end_recording_session, new_state, overlay_state,
    pending_count,
};
use super::worker::{TranscriptionWorkerConfig, transcription_worker};

pub(super) fn run_daemon_loop(
    store: ModelStore,
    options: DaemonOptions,
    overlay: OverlayHandle,
) -> Result<()> {
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
        end_key: matches!(options.hotkey_mode, HotkeyMode::TriggerEnd)
            .then(|| options.end_hotkey.clone())
            .flatten(),
    };
    if matches!(options.hotkey_mode, HotkeyMode::TriggerEnd) && hotkey_cfg.end_key.is_none() {
        bail!("trigger-end 热键模式需要配置 end-hotkey");
    }
    let hotkeys = HotkeyManager::new(&hotkey_cfg)?;
    let state = new_state();

    let (segment_tx, segment_rx) = crossbeam_channel::unbounded();
    let overlay_worker = overlay.clone();
    let state_worker = state.clone();
    let inject_method = options.inject_method.clone();
    let append_newline = options.append_newline;
    let strip_trailing_period = options.strip_trailing_period;
    let idle_unload_secs = options.idle_unload_secs;
    tokio::task::spawn_blocking(move || {
        transcription_worker(
            TranscriptionWorkerConfig {
                store,
                dataset,
                overlay: overlay_worker,
                state: state_worker,
                inject_method,
                append_newline,
                strip_trailing_period,
                idle_unload_secs,
            },
            segment_rx,
        );
    });

    let hotkey_events = HotkeyManager::events().clone();
    let mut capture = CaptureRuntime::new();

    info!(
        hotkey_mode = options.hotkey_mode.label(),
        hotkey = %hotkeys.trigger_label(),
        end_hotkey = hotkeys.end_label().as_deref().unwrap_or(""),
        "daemon 已启动, 等待热键"
    );

    loop {
        crossbeam_channel::select! {
            recv(hotkey_events) -> event => {
                let event = event.context("热键事件通道已关闭")?;
                if let Some(hotkey_event) = hotkeys.action(event) {
                    handle_hotkey_event(
                        hotkey_event,
                        options.hotkey_mode,
                        &mut HotkeyRuntime {
                            capture: &mut capture,
                            segmenter: &mut segmenter,
                            segment_tx: &segment_tx,
                            state: &state,
                            overlay: &overlay,
                            tail_padding_ms: options.tail_padding_ms,
                        },
                    )?;
                }
            }
            default(Duration::from_millis(30)) => {
                if capture.capturing && let Some(rx) = &capture.audio_rx {
                    match rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(frame) => {
                            capture.last_frame_at = Instant::now();
                            let level = frame_level(&frame);
                            let mode = if segmenter.detected() {
                                OverlayMode::Recording { level }
                            } else {
                                OverlayMode::Silence { pending: pending_count(&state) }
                            };
                            overlay.set(overlay_state(&state, mode));
                            for segment in segmenter.push(&frame) {
                                submit_segment(&segment_tx, &state, &overlay, segment)?;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if capture.last_frame_at.elapsed() > Duration::from_millis(options.end_silence_ms as u64) {
                                overlay.set(overlay_state(
                                    &state,
                                    OverlayMode::Silence { pending: pending_count(&state) },
                                ));
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            warn!("音频流断开");
                            capture.capturing = false;
                            end_recording_session(&state);
                            if let Some(input) = capture.audio_input.take() {
                                input.stop();
                            }
                            capture.audio_rx = None;
                        }
                    }
                }
            }
        }
    }
}

struct HotkeyRuntime<'a> {
    capture: &'a mut CaptureRuntime,
    segmenter: &'a mut VadSegmenter,
    segment_tx: &'a Sender<SpeechSegment>,
    state: &'a SharedRuntimeState,
    overlay: &'a OverlayHandle,
    tail_padding_ms: u32,
}

struct CaptureRuntime {
    capturing: bool,
    audio_input: Option<AudioInput>,
    audio_rx: Option<Receiver<Vec<i16>>>,
    last_frame_at: Instant,
}

impl CaptureRuntime {
    fn new() -> Self {
        Self {
            capturing: false,
            audio_input: None,
            audio_rx: None,
            last_frame_at: Instant::now(),
        }
    }
}

fn handle_hotkey_event(
    event: HotkeyEvent,
    mode: HotkeyMode,
    ctx: &mut HotkeyRuntime<'_>,
) -> Result<()> {
    match mode {
        HotkeyMode::Pressed => match event {
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } => begin_capture(ctx.capture, ctx.segmenter, ctx.state, ctx.overlay),
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Released,
            } => end_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.segment_tx,
                ctx.state,
                ctx.overlay,
                ctx.tail_padding_ms,
                "热键释放, 结束录音",
            ),
            _ => Ok(()),
        },
        HotkeyMode::Toggle => match event {
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } if ctx.capture.capturing => end_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.segment_tx,
                ctx.state,
                ctx.overlay,
                ctx.tail_padding_ms,
                "toggle 热键按下, 结束录音",
            ),
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } => begin_capture(ctx.capture, ctx.segmenter, ctx.state, ctx.overlay),
            _ => Ok(()),
        },
        HotkeyMode::TriggerEnd => match event {
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } => begin_capture(ctx.capture, ctx.segmenter, ctx.state, ctx.overlay),
            HotkeyEvent {
                role: HotkeyRole::End,
                action: HotkeyAction::Pressed,
            } => end_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.segment_tx,
                ctx.state,
                ctx.overlay,
                ctx.tail_padding_ms,
                "结束热键按下, 结束录音",
            ),
            _ => Ok(()),
        },
    }
}

fn begin_capture(
    capture: &mut CaptureRuntime,
    segmenter: &mut VadSegmenter,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
) -> Result<()> {
    if capture.capturing {
        return Ok(());
    }
    debug!("开始录音");
    let input = AudioInput::start(None)?;
    capture.audio_rx = Some(input.receiver());
    capture.audio_input = Some(input);
    segmenter.reset();
    capture.capturing = true;
    capture.last_frame_at = Instant::now();
    begin_recording_session(state);
    overlay.set(overlay_state(state, OverlayMode::Recording { level: 0.0 }));
    Ok(())
}

fn end_capture(
    capture: &mut CaptureRuntime,
    segmenter: &mut VadSegmenter,
    segment_tx: &Sender<SpeechSegment>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    tail_padding_ms: u32,
    reason: &str,
) -> Result<()> {
    if !capture.capturing {
        return Ok(());
    }
    debug!("{}", reason);
    capture.capturing = false;
    end_recording_session(state);
    if let Some(rx) = &capture.audio_rx {
        drain_release_tail(
            rx,
            segmenter,
            segment_tx,
            state,
            overlay,
            Duration::from_millis(tail_padding_ms as u64),
        )?;
    }
    for segment in segmenter.finish() {
        submit_segment(segment_tx, state, overlay, segment)?;
    }
    if let Some(input) = capture.audio_input.take() {
        input.stop();
    }
    capture.audio_rx = None;
    let pending = pending_count(state);
    let mode = if pending == 0 {
        OverlayMode::Idle
    } else {
        OverlayMode::Transcribing { pending }
    };
    overlay.set(overlay_state(state, mode));
    Ok(())
}
