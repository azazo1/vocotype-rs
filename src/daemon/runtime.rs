use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use tracing::{debug, info, warn};

use crate::asr::AsrEngine;
use crate::audio::AudioInput;
use crate::dataset::DatasetRecorder;
use crate::hotkey::{HotkeyAction, HotkeyConfig, HotkeyManager, recv_action};
use crate::models::ModelStore;
use crate::overlay::{OverlayHandle, OverlayMode, OverlayState};

use super::DaemonOptions;
use super::segments::{
    build_segmenter, drain_release_tail, frame_level, submit_segment,
};
use super::state::{new_state, pending_count, set_recording};
use super::worker::transcription_worker;

pub(super) fn run_daemon_loop(
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
    let state = new_state();

    let (segment_tx, segment_rx) = crossbeam_channel::unbounded();
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
                        debug!("热键按下, 开始录音");
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
                        debug!("热键释放, 结束录音");
                        capturing = false;
                        set_recording(&state, false);
                        if let Some(rx) = &audio_rx {
                            drain_release_tail(
                                rx,
                                &mut segmenter,
                                &segment_tx,
                                &state,
                                &overlay,
                                Duration::from_millis(options.tail_padding_ms as u64),
                            )?;
                        }
                        for segment in segmenter.finish() {
                            submit_segment(&segment_tx, &state, &overlay, segment)?;
                        }
                        if let Some(input) = audio_input.take() {
                            input.stop();
                        }
                        audio_rx = None;
                        let pending = pending_count(&state);
                        let mode = if pending == 0 {
                            OverlayMode::Idle
                        } else {
                            OverlayMode::Transcribing { pending }
                        };
                        overlay.set(OverlayState { mode });
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
