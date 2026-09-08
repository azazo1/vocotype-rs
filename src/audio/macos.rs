use std::ptr::NonNull;
use std::sync::{
    Arc,
    atomic::AtomicU64,
};

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
use tracing::info;

use crate::asr::TARGET_SAMPLE_RATE;

use super::{f32_to_i16, send_audio_frame};

const INPUT_BUS: usize = 0;
const TAP_BUFFER_SIZE: u32 = 1_024;

type InputTap = RcBlock<dyn Fn(NonNull<AVAudioPCMBuffer>, NonNull<AVAudioTime>)>;

pub(super) struct VoiceProcessingInput {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    _tap: InputTap,
    tap_installed: bool,
}

impl VoiceProcessingInput {
    pub(super) fn start() -> Result<(Self, Receiver<Vec<i16>>)> {
        let engine = unsafe { AVAudioEngine::init(AVAudioEngine::alloc()) };
        let input = unsafe { engine.inputNode() };
        unsafe { input.setVoiceProcessingEnabled_error(true) }
            .map_err(|error| anyhow!(error.to_string()))
            .context("无法启用 Voice Processing IO")?;
        unsafe {
            input.setVoiceProcessingOtherAudioDuckingConfiguration(
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration {
                    enableAdvancedDucking: Bool::NO,
                    duckingLevel: AVAudioVoiceProcessingOtherAudioDuckingLevel::Default,
                },
            );
            input.setVoiceProcessingInputMuted(false);
        }

        let mixer = unsafe { engine.mainMixerNode() };
        unsafe {
            mixer.setOutputVolume(0.0);
            engine.connect_to_format(&input, &mixer, None);
            engine.prepare();
        }
        if let Err(error) = unsafe { engine.startAndReturnError() } {
            return Err(anyhow!(error.to_string())).context("无法启动 AVAudioEngine");
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

        let mut stream = Self {
            engine,
            input,
            _tap: tap,
            tap_installed: false,
        };
        unsafe {
            stream.input.installTapOnBus_bufferSize_format_block(
                INPUT_BUS,
                TAP_BUFFER_SIZE,
                None,
                RcBlock::as_ptr(&stream._tap),
            );
        }
        stream.tap_installed = true;

        info!(
            input_sample_rate,
            target_sample_rate = TARGET_SAMPLE_RATE,
            channels,
            interleaved,
            voice_processing_input_muted = muted,
            ducking_level = "default",
            advanced_ducking = false,
            "macOS 语音处理音频采集已启动"
        );
        Ok((stream, receiver))
    }
}

impl Drop for VoiceProcessingInput {
    fn drop(&mut self) {
        if self.tap_installed {
            unsafe {
                self.input.removeTapOnBus(INPUT_BUS);
            }
            self.tap_installed = false;
        }
        unsafe {
            self.engine.stop();
        }
        info!("macOS 语音处理音频采集已停止, 其他应用音量已恢复");
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
