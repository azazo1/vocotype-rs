use std::path::Path;

use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct PcmAudio {
    pub sample_rate: u32,
    pub samples: Vec<i16>,
}

impl PcmAudio {
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

pub fn read_wav_mono_i16(path: &Path, target_sample_rate: u32) -> Result<PcmAudio> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("无法打开音频文件: {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        bail!("音频文件没有声道: {}", path.display());
    }

    let raw = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                reader
                    .samples::<i16>()
                    .collect::<Result<Vec<_>, _>>()
                    .with_context(|| format!("无法读取 i16 音频样本: {}", path.display()))?
            } else {
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| (value >> 16).clamp(i16::MIN as i32, i16::MAX as i32) as i16))
                    .collect::<Result<Vec<_>, _>>()
                    .with_context(|| format!("无法读取 i32 音频样本: {}", path.display()))?
            }
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map(|value| {
                    let value = value.clamp(-1.0, 1.0);
                    (value * i16::MAX as f32) as i16
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("无法读取 f32 音频样本: {}", path.display()))?,
    };

    let mono = downmix_to_mono(&raw, spec.channels as usize);
    let samples = resample_linear_i16(&mono, spec.sample_rate, target_sample_rate);
    Ok(PcmAudio {
        sample_rate: target_sample_rate,
        samples,
    })
}

pub fn write_wav_mono_i16(path: &Path, sample_rate: u32, samples: &[i16]) -> Result<()> {
    crate::app::ensure_parent(path)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("无法创建音频文件: {}", path.display()))?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn downmix_to_mono(samples: &[i16], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|frame| {
            let sum = frame.iter().map(|value| *value as i32).sum::<i32>();
            (sum / frame.len() as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect()
}

pub fn resample_linear_i16(samples: &[i16], input_rate: u32, target_rate: u32) -> Vec<i16> {
    if samples.is_empty() || input_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = input_rate as f64 / target_rate as f64;
    let out_len = (samples.len() as f64 / ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);

    for idx in 0..out_len {
        let pos = idx as f64 * ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = pos - lo as f64;
        let sample = samples[lo] as f64 * (1.0 - frac) + samples[hi] as f64 * frac;
        out.push(sample.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
    }

    out
}
