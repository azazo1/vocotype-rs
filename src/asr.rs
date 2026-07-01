use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use sherpa_onnx::{OfflineParaformerModelConfig, OfflineRecognizer, OfflineRecognizerConfig};
use tracing::{debug, info};

use crate::models::{AsrModelFiles, ModelStore};
use crate::wav::PcmAudio;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug, Serialize)]
pub struct TranscriptionResult {
    pub success: bool,
    pub text: String,
    pub raw_text: String,
    pub duration: f32,
    pub inference_latency: f32,
    pub confidence: f32,
    pub error: Option<String>,
}

pub struct AsrEngine {
    recognizer: OfflineRecognizer,
}

impl AsrEngine {
    pub fn load(store: ModelStore) -> Result<Arc<Self>> {
        store.verify_required()?;
        let files = store.asr_model_files()?;
        let recognizer = create_recognizer(&files)?;
        info!(
            model = %files.model.display(),
            tokens = %files.tokens.display(),
            "ASR 模型加载完成"
        );
        Ok(Arc::new(Self { recognizer }))
    }

    pub fn transcribe_file(&self, audio: &Path) -> Result<TranscriptionResult> {
        let pcm = crate::wav::read_wav_mono_i16(audio, TARGET_SAMPLE_RATE)?;
        self.transcribe_pcm(&pcm)
    }

    pub fn transcribe_pcm(&self, audio: &PcmAudio) -> Result<TranscriptionResult> {
        let start = Instant::now();
        if audio.samples.is_empty() {
            bail!("音频为空, 跳过转写");
        }

        let samples_i16 = if audio.sample_rate == TARGET_SAMPLE_RATE {
            audio.samples.clone()
        } else {
            crate::wav::resample_linear_i16(&audio.samples, audio.sample_rate, TARGET_SAMPLE_RATE)
        };
        let samples = i16_to_f32(&samples_i16);
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, &samples);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("ASR 解码没有返回结果"))?;
        let text = result.text.trim().to_string();
        let latency = start.elapsed().as_secs_f32();
        debug!(
            duration = audio.duration_seconds(),
            latency,
            chars = text.chars().count(),
            "ASR 转写完成"
        );

        let success = !text.is_empty();
        Ok(TranscriptionResult {
            success,
            text: text.clone(),
            raw_text: text,
            duration: audio.duration_seconds(),
            inference_latency: latency,
            confidence: if success { 1.0 } else { 0.0 },
            error: if success {
                None
            } else {
                Some("没有识别到文本".to_string())
            },
        })
    }
}

fn create_recognizer(files: &AsrModelFiles) -> Result<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.paraformer = OfflineParaformerModelConfig {
        model: Some(path_string(&files.model)?),
    };
    config.model_config.tokens = Some(path_string(&files.tokens)?);
    config.model_config.num_threads = 2;
    config.model_config.provider = Some("cpu".to_string());

    OfflineRecognizer::create(&config).with_context(|| {
        format!(
            "无法加载 sherpa ASR 模型: model={}, tokens={}",
            files.model.display(),
            files.tokens.display()
        )
    })
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("路径不是有效 UTF-8: {}", path.display()))
}

fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| *sample as f32 / i16::MAX as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_i16_to_normalized_f32() {
        let samples = i16_to_f32(&[0, i16::MAX, i16::MIN]);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 1.0).abs() < 0.0001);
        assert!(samples[2] <= -1.0);
    }
}
