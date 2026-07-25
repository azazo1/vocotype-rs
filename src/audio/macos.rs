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
        }

        let format = unsafe { input.outputFormatForBus(INPUT_BUS) };
        if unsafe { format.commonFormat() } != AVAudioCommonFormat::PCMFormatFloat32 {
            bail!("Voice Processing IO 没有提供 float32 PCM 输入");
        }
        let sample_rate = unsafe { format.sampleRate() };
        let channels = unsafe { format.channelCount() } as usize;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            bail!("Voice Processing IO 返回了无效采样率: {sample_rate}");
        }
        if channels == 0 {
            bail!("Voice Processing IO 没有可用输入声道");
        }
        let input_sample_rate = sample_rate.round() as u32;
        let (sender, receiver) = bounded::<Vec<i16>>(256);
        let dropped_frames = Arc::new(AtomicU64::new(0));
        let tap: InputTap = RcBlock::new({
            let dropped_frames = dropped_frames.clone();
            move |buffer: NonNull<AVAudioPCMBuffer>, _time: NonNull<AVAudioTime>| {
                let buffer = unsafe { buffer.as_ref() };
                let Some(mono) = pcm_buffer_to_mono_i16(buffer, channels) else {
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
                Some(&format),
                RcBlock::as_ptr(&stream._tap),
            );
        }
        stream.tap_installed = true;
        unsafe {
            stream.engine.prepare();
        }
        if let Err(error) = unsafe { stream.engine.startAndReturnError() } {
            drop(stream);
            return Err(anyhow!(error.to_string())).context("无法启动 AVAudioEngine");
        }

        info!(
            input_sample_rate,
            target_sample_rate = TARGET_SAMPLE_RATE,
            channels,
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

fn pcm_buffer_to_mono_i16(buffer: &AVAudioPCMBuffer, channels: usize) -> Option<Vec<i16>> {
    let frame_count = unsafe { buffer.frameLength() } as usize;
    if frame_count == 0 {
        return Some(Vec::new());
    }
    let stride = unsafe { buffer.stride() };
    let channel_data = unsafe { buffer.floatChannelData() };
    if channel_data.is_null() || stride == 0 {
        return None;
    }

    let pointers = unsafe { std::slice::from_raw_parts(channel_data, channels) };
    Some(downmix_float_channels(pointers, frame_count, stride))
}

fn downmix_float_channels(
    channels: &[NonNull<f32>],
    frame_count: usize,
    stride: usize,
) -> Vec<i16> {
    let mut mono = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let sum = channels
            .iter()
            .map(|channel| unsafe { *channel.as_ptr().add(frame * stride) })
            .sum::<f32>();
        mono.push(f32_to_i16(sum / channels.len() as f32));
    }
    mono
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_and_converts_float_channels() {
        let left = [1.0, 0.5, -1.0];
        let right = [-1.0, 0.5, 1.0];
        let channels = [
            NonNull::new(left.as_ptr().cast_mut()).unwrap(),
            NonNull::new(right.as_ptr().cast_mut()).unwrap(),
        ];

        let mono = downmix_float_channels(&channels, 3, 1);

        assert_eq!(mono, vec![0, i16::MAX / 2, 0]);
    }

    #[test]
    fn resamples_voice_processing_audio_to_target_rate() {
        let source = vec![i16::MAX; 480];

        let output = crate::wav::resample_linear_i16(&source, 48_000, TARGET_SAMPLE_RATE);

        assert_eq!(output.len(), 160);
        assert!(output.iter().all(|sample| *sample == i16::MAX));
    }
}
