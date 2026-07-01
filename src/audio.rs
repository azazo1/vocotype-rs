use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use crossbeam_channel::{Receiver, bounded};
use tracing::{info, warn};

use crate::asr::TARGET_SAMPLE_RATE;

pub struct AudioInput {
    stream: Stream,
    receiver: Receiver<Vec<i16>>,
}

impl AudioInput {
    pub fn start(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_input_device(&host, device_name)?;
        let device_label = device.name().unwrap_or_else(|_| "unknown".to_string());
        let supported = device
            .default_input_config()
            .with_context(|| format!("无法读取输入设备配置: {}", device_label))?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let input_sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        let (sender, receiver) = bounded::<Vec<i16>>(256);

        let err_fn = |error| warn!(%error, "音频输入流错误");
        let stream = match sample_format {
            SampleFormat::F32 => {
                let sender = sender.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        let mono = f32_to_mono_i16(data, channels);
                        let resampled = crate::wav::resample_linear_i16(
                            &mono,
                            input_sample_rate,
                            TARGET_SAMPLE_RATE,
                        );
                        let _ = sender.try_send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let sender = sender.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        let mono = crate::wav::downmix_to_mono(data, channels);
                        let resampled = crate::wav::resample_linear_i16(
                            &mono,
                            input_sample_rate,
                            TARGET_SAMPLE_RATE,
                        );
                        let _ = sender.try_send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let sender = sender.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        let mono = u16_to_mono_i16(data, channels);
                        let resampled = crate::wav::resample_linear_i16(
                            &mono,
                            input_sample_rate,
                            TARGET_SAMPLE_RATE,
                        );
                        let _ = sender.try_send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            other => bail!("不支持的音频采样格式: {:?}", other),
        };

        stream.play()?;
        info!(
            device = %device_label,
            input_sample_rate,
            target_sample_rate = TARGET_SAMPLE_RATE,
            channels,
            "音频采集已启动"
        );

        Ok(Self { stream, receiver })
    }

    pub fn receiver(&self) -> Receiver<Vec<i16>> {
        self.receiver.clone()
    }

    pub fn stop(self) {
        drop(self.stream);
        info!("音频采集已停止");
    }
}

pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;
    Ok(devices.filter_map(|device| device.name().ok()).collect())
}

fn select_input_device(host: &cpal::Host, device_name: Option<&str>) -> Result<cpal::Device> {
    if let Some(expected) = device_name {
        for device in host.input_devices()? {
            let name = device.name().unwrap_or_default();
            if name.contains(expected) {
                return Ok(device);
            }
        }
        return Err(anyhow!("没有找到输入设备: {}", expected));
    }

    host.default_input_device()
        .ok_or_else(|| anyhow!("没有可用的默认输入设备"))
}

fn f32_to_mono_i16(data: &[f32], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return data.iter().map(|value| f32_to_i16(*value)).collect();
    }
    data.chunks(channels)
        .map(|frame| {
            let sum = frame.iter().copied().sum::<f32>();
            f32_to_i16(sum / frame.len() as f32)
        })
        .collect()
}

fn u16_to_mono_i16(data: &[u16], channels: usize) -> Vec<i16> {
    let converted = data
        .iter()
        .map(|value| (*value as i32 - 32_768).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
        .collect::<Vec<_>>();
    crate::wav::downmix_to_mono(&converted, channels)
}

fn f32_to_i16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}
