use std::ptr::NonNull;
use std::sync::{
    Arc,
    atomic::AtomicU64,
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use block2::RcBlock;
use crossbeam_channel::{Receiver, bounded};
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2::AnyThread;
use objc2_avf_audio::{
    AVAudioCommonFormat, AVAudioEngine, AVAudioInputNode, AVAudioPCMBuffer, AVAudioTime,
    AVAudioVoiceProcessingOtherAudioDuckingConfiguration,
    AVAudioVoiceProcessingOtherAudioDuckingLevel,
};
use tracing::{debug, info, warn};

use crate::asr::TARGET_SAMPLE_RATE;

use super::{f32_to_i16, send_audio_frame};

const INPUT_BUS: usize = 0;
const TAP_BUFFER_SIZE: u32 = 1_024;
const RESTART_BACKOFF: Duration = Duration::from_millis(500);

type InputTap = RcBlock<dyn Fn(NonNull<AVAudioPCMBuffer>, NonNull<AVAudioTime>)>;

pub(super) struct VoiceProcessingInput {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    ducking: AVAudioVoiceProcessingOtherAudioDuckingConfiguration,
    _tap: InputTap,
    restarts: u64,
    failures: u64,
    next_attempt_at: Instant,
}

impl VoiceProcessingInput {
    pub(super) fn start() -> Result<(Self, Receiver<Vec<i16>>)> {
        let started_at = Instant::now();
        let engine = unsafe { AVAudioEngine::init(AVAudioEngine::alloc()) };
        let input = unsafe { engine.inputNode() };

        // 装配顺序不能调换: 先让引擎按输出设备采样率建立输出侧图, 再启用 Voice Processing.
        // 反过来时 Voice Processing 会把 outputNode 的输入格式改成 44100, 而输入节点是
        // 48000/3ch, 两者不一致会让 engine.start() 以 -10875 失败, 采集只能退回 CPAL,
        // 结果就是引擎启动慢一截, 而且监听期间 ducking 完全失效.
        let mixer = unsafe { engine.mainMixerNode() };
        unsafe {
            mixer.setOutputVolume(0.0);
            engine.connect_to_format(&input, &mixer, None);
        }

        let ducking = AVAudioVoiceProcessingOtherAudioDuckingConfiguration {
            enableAdvancedDucking: Bool::NO,
            duckingLevel: AVAudioVoiceProcessingOtherAudioDuckingLevel::Default,
        };
        unsafe { input.setVoiceProcessingEnabled_error(true) }
            .map_err(|error| anyhow!(error.to_string()))
            .context("无法启用 Voice Processing IO")?;
        unsafe {
            input.setVoiceProcessingOtherAudioDuckingConfiguration(ducking);
            input.setVoiceProcessingInputMuted(false);
        }

        let format = unsafe { input.outputFormatForBus(INPUT_BUS) };
        if let Err(error) = validate_voice_processing_format(&format) {
            unsafe {
                engine.stop();
            }
            return Err(error);
        }
        let sample_rate = unsafe { format.sampleRate() };
        let channels = unsafe { format.channelCount() } as usize;
        let input_sample_rate = sample_rate.round() as u32;
        let interleaved = unsafe { format.isInterleaved() };
        let muted = unsafe { input.isVoiceProcessingInputMuted() };
        let (sender, receiver) = bounded::<Vec<i16>>(256);
        let dropped_frames = Arc::new(AtomicU64::new(0));
        let tap: InputTap = RcBlock::new({
            let dropped_frames = dropped_frames.clone();
            move |buffer: NonNull<AVAudioPCMBuffer>, _time: NonNull<AVAudioTime>| {
                let buffer = unsafe { buffer.as_ref() };
                let Some(mono) = pcm_buffer_to_mono_i16(buffer) else {
                    return;
                };
                let resampled = crate::wav::resample_linear_i16(
                    &mono,
                    input_sample_rate,
                    TARGET_SAMPLE_RATE,
                );
                send_audio_frame(&sender, resampled, &dropped_frames);
            }
        });

        let stream = Self {
            engine,
            input,
            ducking,
            _tap: tap,
            restarts: 0,
            failures: 0,
            next_attempt_at: Instant::now(),
        };
        unsafe {
            stream.input.installTapOnBus_bufferSize_format_block(
                INPUT_BUS,
                TAP_BUFFER_SIZE,
                None,
                RcBlock::as_ptr(&stream._tap),
            );
        }
        unsafe {
            stream.engine.prepare();
        }
        if let Err(error) = unsafe { stream.engine.startAndReturnError() } {
            return Err(anyhow!(error.to_string())).context("无法启动 AVAudioEngine");
        }

        info!(
            input_sample_rate,
            target_sample_rate = TARGET_SAMPLE_RATE,
            channels,
            interleaved,
            voice_processing_input_muted = muted,
            ducking_level = "default",
            advanced_ducking = false,
            startup_ms = started_at.elapsed().as_millis() as u64,
            "macOS 语音处理音频采集已启动"
        );
        Ok((stream, receiver))
    }

    /// Voice Processing 的引擎会被系统的音频配置变更停掉(例如切换设备, 采样率变化, 通话抢占),
    /// 引擎一停, 采集和 ducking 就一起失效, 所以采集期间需要不断自检并把引擎重新拉起来.
    /// 由 daemon 在采集轮询里定期调用.
    pub(super) fn maintain(&mut self) {
        if unsafe { self.engine.isRunning() } {
            self.failures = 0;
            return;
        }
        let now = Instant::now();
        if now < self.next_attempt_at {
            return;
        }
        match restart_engine(&self.engine, &self.input, self.ducking) {
            Ok(()) => {
                self.restarts += 1;
                self.failures = 0;
                self.next_attempt_at = now;
                if self.restarts <= 3 {
                    info!(restarts = self.restarts, "音频引擎被系统重新配置, 已自动重启");
                } else {
                    debug!(restarts = self.restarts, "音频引擎已自动重启");
                }
            }
            Err(error) => {
                self.failures += 1;
                self.next_attempt_at = now + RESTART_BACKOFF;
                if self.failures <= 3 || self.failures.is_multiple_of(20) {
                    warn!(failures = self.failures, "{}", error);
                }
            }
        }
    }
}

impl Drop for VoiceProcessingInput {
    fn drop(&mut self) {
        unsafe {
            self.input.removeTapOnBus(INPUT_BUS);
            self.engine.stop();
        }
        info!("macOS 语音处理音频采集已停止, 其他应用音量已恢复");
    }
}

fn restart_engine(
    engine: &AVAudioEngine,
    input: &AVAudioInputNode,
    ducking: AVAudioVoiceProcessingOtherAudioDuckingConfiguration,
) -> Result<(), String> {
    unsafe {
        if !input.isVoiceProcessingEnabled() {
            input
                .setVoiceProcessingEnabled_error(true)
                .map_err(|error| format!("无法重新启用 Voice Processing IO: {}", error))?;
        }
        input.setVoiceProcessingOtherAudioDuckingConfiguration(ducking);
        input.setVoiceProcessingInputMuted(false);
        engine.prepare();
        engine
            .startAndReturnError()
            .map_err(|error| format!("无法启动 AVAudioEngine: {}", error))
    }
}

fn validate_voice_processing_format(
    format: &objc2_avf_audio::AVAudioFormat,
) -> Result<()> {
    if unsafe { format.commonFormat() } != AVAudioCommonFormat::PCMFormatFloat32 {
        bail!("Voice Processing IO 没有提供 float32 PCM 输入");
    }
    let sample_rate = unsafe { format.sampleRate() };
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        bail!("Voice Processing IO 返回了无效采样率: {sample_rate}");
    }
    if unsafe { format.channelCount() } == 0 {
        bail!("Voice Processing IO 没有可用输入声道");
    }
    Ok(())
}

fn pcm_buffer_to_mono_i16(buffer: &AVAudioPCMBuffer) -> Option<Vec<i16>> {
    let frame_count = unsafe { buffer.frameLength() } as usize;
    if frame_count == 0 {
        return Some(Vec::new());
    }
    let format = unsafe { buffer.format() };
    let channels = unsafe { format.channelCount() } as usize;
    let stride = unsafe { buffer.stride() };
    let channel_data = unsafe { buffer.floatChannelData() };
    if channel_data.is_null() || stride == 0 || channels == 0 {
        return None;
    }

    let first_channel = unsafe { *channel_data };
    Some(first_channel_to_i16(first_channel, frame_count, stride))
}

fn first_channel_to_i16(
    channel: NonNull<f32>,
    frame_count: usize,
    stride: usize,
) -> Vec<i16> {
    let mut mono = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let sample = unsafe { *channel.as_ptr().add(frame * stride) };
        mono.push(f32_to_i16(sample));
    }
    mono
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_first_channel_without_averaging() {
        let primary = [1.0, 0.5, -1.0];
        let channel = NonNull::new(primary.as_ptr().cast_mut()).unwrap();

        let mono = first_channel_to_i16(channel, 3, 1);

        assert_eq!(mono, vec![i16::MAX, i16::MAX / 2, -i16::MAX]);
    }

    #[test]
    fn reads_interleaved_first_channel_with_stride() {
        let interleaved = [1.0, -1.0, 0.5, -0.5, -1.0, 1.0];
        let channel = NonNull::new(interleaved.as_ptr().cast_mut()).unwrap();

        let mono = first_channel_to_i16(channel, 3, 2);

        assert_eq!(mono, vec![i16::MAX, i16::MAX / 2, -i16::MAX]);
    }

    #[test]
    fn resamples_voice_processing_audio_to_target_rate() {
        let source = vec![i16::MAX; 480];

        let output = crate::wav::resample_linear_i16(&source, 48_000, TARGET_SAMPLE_RATE);

        assert_eq!(output.len(), 160);
        assert!(output.iter().all(|sample| *sample == i16::MAX));
    }
}
