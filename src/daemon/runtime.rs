use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use tracing::{debug, info, warn};

use crate::asr_backend::AsrBackend;
use crate::audio::AudioInput;
use crate::dataset::DatasetRecorder;
use crate::config::DuckOtherAudioPreference;
use crate::hotkey::{HotkeyAction, HotkeyEvent, HotkeyManager, HotkeyMatcher, HotkeyRole};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode};
use crate::vad::VadSegmenter;

use super::{DaemonOptions, HotkeyMode};
use super::segments::{
    build_segmenter, frame_level, submit_segment, submit_stream_audio, submit_stream_finish,
    submit_stream_start,
};
use super::state::{
    SharedRuntimeState, begin_recording_session, end_recording_session, new_state, overlay_state,
    pending_count,
};
use super::worker::{TranscriptionTask, TranscriptionWorkerConfig, transcription_worker};

pub(super) fn run_daemon_loop(
    store: ModelStore,
    options: DaemonOptions,
    overlay: OverlayHandle,
    hotkeys: HotkeyMatcher,
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

    let state = new_state();

    let (task_tx, task_rx) = crossbeam_channel::unbounded();
    let overlay_worker = overlay.clone();
    let state_worker = state.clone();
    let inject_method = options.inject_method.clone();
    let append_newline = options.append_newline;
    let idle_unload_secs = options.idle_unload_secs;
    let asr_options = options.asr_options.clone();
    tokio::task::spawn_blocking(move || {
        transcription_worker(
            TranscriptionWorkerConfig {
                store,
                dataset,
                overlay: overlay_worker,
                state: state_worker,
                inject_method,
                append_newline,
                idle_unload_secs,
                asr_options,
            },
            task_rx,
        );
    });

    let hotkey_events = HotkeyManager::events().clone();
    let mut capture = CaptureRuntime::new(options.asr_options.backend == AsrBackend::Iflytek);

    info!(
        hotkey_mode = options.hotkey_mode.label(),
        hotkey = %hotkeys.trigger_label(),
        end_hotkey = hotkeys.end_label().as_deref().unwrap_or(""),
        duck_other_audio = options.duck_other_audio.enabled(),
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
                            task_tx: &task_tx,
                            state: &state,
                            overlay: &overlay,
                            tail_padding_ms: options.tail_padding_ms,
                            duck_other_audio: &options.duck_other_audio,
                        },
                    )?;
                }
            }
            default(Duration::from_millis(30)) => {
                if capture.capturing && let Some(rx) = capture.audio_rx.clone() {
                    match rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(frame) => {
                            process_audio_frame(
                                frame,
                                &mut segmenter,
                                &task_tx,
                                &state,
                                &overlay,
                                &mut capture,
                            )?;
                            drain_queued_audio_frames(
                                &rx,
                                &mut segmenter,
                                &task_tx,
                                &state,
                                &overlay,
                                &mut capture,
                            )?;
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if capture.last_frame_at.elapsed() > Duration::from_millis(options.end_silence_ms as u64) {
                                report_capture_silence(&mut capture, &state, &overlay);
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            warn!("音频流断开");
                            end_capture(
                                &mut capture,
                                &mut segmenter,
                                &task_tx,
                                &state,
                                &overlay,
                                options.tail_padding_ms,
                                "音频流断开, 结束录音",
                            )?;
                        }
                    }
                }
            }
        }
    }
}

fn process_audio_frame(
    frame: Vec<i16>,
    segmenter: &mut Option<VadSegmenter>,
    task_tx: &Sender<TranscriptionTask>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    capture: &mut CaptureRuntime,
) -> Result<()> {
    capture.last_frame_at = Instant::now();
    capture.received_frames = capture.received_frames.saturating_add(1);
    if capture.received_frames == 1 {
        info!(
            samples = frame.len(),
            level = frame_level(&frame),
            "收到首帧麦克风音频"
        );
    }
    let level = frame_level(&frame);
    if capture.streaming_backend {
        if capture.stream_active {
            submit_stream_audio(task_tx, frame)?;
        }
        return Ok(());
    }

    let segmenter = segmenter
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("sherpa VAD 分段器未初始化"))?;
    let mode = if segmenter.detected() {
        OverlayMode::Recording { level }
    } else {
        OverlayMode::Silence {
            pending: pending_count(state),
        }
    };
    overlay.set(overlay_state(state, mode));
    let segments = segmenter.push(&frame)?;
    for segment in segments {
        submit_segment(task_tx, state, overlay, segment)?;
    }
    Ok(())
}

fn drain_queued_audio_frames(
    audio_rx: &Receiver<Vec<i16>>,
    segmenter: &mut Option<VadSegmenter>,
    task_tx: &Sender<TranscriptionTask>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    capture: &mut CaptureRuntime,
) -> Result<()> {
    loop {
        match audio_rx.try_recv() {
            Ok(frame) => {
                process_audio_frame(frame, segmenter, task_tx, state, overlay, capture)?
            }
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Disconnected) => {
                warn!("音频流断开");
                return Ok(());
            }
        }
    }
}

struct HotkeyRuntime<'a> {
    capture: &'a mut CaptureRuntime,
    segmenter: &'a mut Option<VadSegmenter>,
    task_tx: &'a Sender<TranscriptionTask>,
    state: &'a SharedRuntimeState,
    overlay: &'a OverlayHandle,
    tail_padding_ms: u32,
    duck_other_audio: &'a DuckOtherAudioPreference,
}

struct CaptureRuntime {
    capturing: bool,
    audio_input: Option<AudioInput>,
    audio_rx: Option<Receiver<Vec<i16>>>,
    last_frame_at: Instant,
    received_frames: u64,
    missing_input_logged: bool,
    streaming_backend: bool,
    stream_active: bool,
}

impl CaptureRuntime {
    fn new(streaming_backend: bool) -> Self {
        Self {
            capturing: false,
            audio_input: None,
            audio_rx: None,
            last_frame_at: Instant::now(),
            received_frames: 0,
            missing_input_logged: false,
            streaming_backend,
            stream_active: false,
        }
    }
}

fn report_capture_silence(
    capture: &mut CaptureRuntime,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
) {
    if capture.received_frames == 0 {
        if !capture.missing_input_logged {
            capture.missing_input_logged = true;
            warn!("录音已启动但没有收到麦克风数据");
        }
        overlay.set(overlay_state(
            state,
            OverlayMode::Error {
                message: "没有收到麦克风数据".to_string(),
            },
        ));
        return;
    }
    overlay.set(overlay_state(
        state,
        OverlayMode::Silence {
            pending: pending_count(state),
        },
    ));
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
            } => begin_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.task_tx,
                ctx.state,
                ctx.overlay,
                ctx.duck_other_audio.enabled(),
            ),
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Released,
            } => end_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.task_tx,
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
                ctx.task_tx,
                ctx.state,
                ctx.overlay,
                ctx.tail_padding_ms,
                "toggle 热键按下, 结束录音",
            ),
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } => begin_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.task_tx,
                ctx.state,
                ctx.overlay,
                ctx.duck_other_audio.enabled(),
            ),
            _ => Ok(()),
        },
        HotkeyMode::TriggerEnd => match event {
            HotkeyEvent {
                role: HotkeyRole::Trigger,
                action: HotkeyAction::Pressed,
            } => begin_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.task_tx,
                ctx.state,
                ctx.overlay,
                ctx.duck_other_audio.enabled(),
            ),
            HotkeyEvent {
                role: HotkeyRole::End,
                action: HotkeyAction::Pressed,
            } => end_capture(
                ctx.capture,
                ctx.segmenter,
                ctx.task_tx,
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
    segmenter: &mut Option<VadSegmenter>,
    task_tx: &Sender<TranscriptionTask>,
    state: &SharedRuntimeState,
    overlay: &OverlayHandle,
    duck_other_audio: bool,
) -> Result<()> {
    if capture.capturing {
        return Ok(());
    }
    debug!("开始录音");
    begin_recording_session(state);
    overlay.set(overlay_state(state, OverlayMode::Starting));
    let input = AudioInput::start(None, duck_other_audio).map_err(|error| {
        overlay.set(overlay_state(
            state,
            OverlayMode::Error {
                message: format!("音频采集启动失败: {}", error),
            },
        ));
        error
    })?;
    capture.audio_rx = Some(input.receiver());
    capture.audio_input = Some(input);
    if let Some(segmenter) = segmenter {
        segmenter.reset();
    }
    capture.stream_active = if capture.streaming_backend {
        submit_stream_start(task_tx, state, overlay, Vec::new())?
    } else {
        false
    };
    capture.capturing = true;
    capture.received_frames = 0;
    capture.missing_input_logged = false;
    capture.last_frame_at = Instant::now();
    overlay.set(overlay_state(state, OverlayMode::Recording { level: 0.0 }));
    Ok(())
}

fn end_capture(
    capture: &mut CaptureRuntime,
    segmenter: &mut Option<VadSegmenter>,
    task_tx: &Sender<TranscriptionTask>,
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
    if let Some(rx) = capture.audio_rx.clone() {
        let deadline = Instant::now() + Duration::from_millis(tail_padding_ms as u64);
        while Instant::now() < deadline {
            let timeout = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10));
            match rx.recv_timeout(timeout) {
                Ok(frame) => {
                    process_audio_frame(frame, segmenter, task_tx, state, overlay, capture)?;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }
    }
    if capture.streaming_backend {
        if capture.stream_active {
            submit_stream_finish(task_tx)?;
            capture.stream_active = false;
        }
    } else if let Some(segmenter) = segmenter {
        for segment in segmenter.finish()? {
            submit_segment(task_tx, state, overlay, segment)?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iflytek_capture_starts_without_an_external_vad_segment() {
        let capture = CaptureRuntime::new(true);
        assert!(capture.streaming_backend);
        assert!(!capture.stream_active);
    }
}
