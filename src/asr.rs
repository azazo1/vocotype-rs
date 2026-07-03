use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use sherpa_onnx::{
    OfflineParaformerModelConfig, OfflinePunctuation, OfflinePunctuationConfig, OfflineRecognizer,
    OfflineRecognizerConfig,
};
use tracing::{debug, info, warn};

use crate::dict::{DEFAULT_HOTWORDS_SCORE, SpeechDictionary};
use crate::models::{AsrModelFiles, ModelStore, PuncModelFiles};
use crate::punctuation::{convert_to_english_punctuation, strip_trailing_period};
use crate::wav::PcmAudio;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const EMPTY_TRANSCRIPTION_MESSAGE: &str = "没有识别到文本";

#[derive(Clone, Debug, Serialize)]
pub struct TranscriptionResult {
    pub success: bool,
    pub text: String,
    pub raw_text: String,
    pub tokens: Vec<String>,
    pub token_timestamps: Option<Vec<f32>>,
    pub duration: f32,
    pub inference_latency: f32,
    pub confidence: f32,
    pub error: Option<String>,
}

impl TranscriptionResult {
    pub fn is_empty_transcription(&self) -> bool {
        !self.success
            && self.text.trim().is_empty()
            && self.error.as_deref() == Some(EMPTY_TRANSCRIPTION_MESSAGE)
    }
}

pub struct AsrEngine {
    recognizer: OfflineRecognizer,
    punctuator: OfflinePunctuation,
    dictionary: SpeechDictionary,
    english_punctuation: bool,
    strip_trailing_period: bool,
}

#[derive(Clone, Debug)]
pub struct AsrOptions {
    pub dictionary: SpeechDictionary,
    pub hotwords_score: f32,
    pub english_punctuation: bool,
    pub strip_trailing_period: bool,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            dictionary: SpeechDictionary::builtin(),
            hotwords_score: DEFAULT_HOTWORDS_SCORE,
            english_punctuation: false,
            strip_trailing_period: false,
        }
    }
}

impl AsrEngine {
    pub fn load(store: ModelStore) -> Result<Arc<Self>> {
        Self::load_with_options(store, AsrOptions::default())
    }

    pub fn load_with_options(store: ModelStore, options: AsrOptions) -> Result<Arc<Self>> {
        store.verify_required()?;
        let asr_files = store.asr_model_files()?;
        let punc_files = store.punc_model_files()?;
        let recognizer = create_recognizer(&asr_files, options.hotwords_score)?;
        let punctuator = create_punctuator(&punc_files)?;
        info!(
            model = %asr_files.model.display(),
            tokens = %asr_files.tokens.display(),
            hotwords = options.dictionary.hotword_count(),
            hotword_rewrites = options.dictionary.hotword_rewrite_count(),
            english_punctuation = options.english_punctuation,
            strip_trailing_period = options.strip_trailing_period,
            "ASR 模型加载完成"
        );
        if options.dictionary.hotword_count() > 0 {
            info!("ASR 模型不支持 sherpa contextual biasing, 使用词表后处理");
        }
        info!(
            model = %punc_files.model.display(),
            "PUNC 模型加载完成"
        );
        Ok(Arc::new(Self {
            recognizer,
            punctuator,
            dictionary: options.dictionary,
            english_punctuation: options.english_punctuation,
            strip_trailing_period: options.strip_trailing_period,
        }))
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
        let raw_text = result.text.trim().to_string();
        let tokens = result.tokens;
        let token_timestamps = result.timestamps;
        let latency = start.elapsed().as_secs_f32();
        let duration = audio.duration_seconds();
        let duration_label = format!("{:.2}", duration);
        let latency_label = format!("{:.2}", latency);
        let success = !raw_text.is_empty();
        let text = if success {
            let punctuated = self.restore_punctuation(&raw_text);
            let mut rewritten = self.dictionary.rewrite_text(&punctuated);
            if self.english_punctuation {
                rewritten = convert_to_english_punctuation(&rewritten).unwrap_or(rewritten);
            }
            if self.strip_trailing_period {
                strip_trailing_period(&rewritten)
            } else {
                rewritten
            }
        } else {
            raw_text.clone()
        };
        if success {
            info!(
                duration = %duration_label,
                latency = %latency_label,
                chars = raw_text.chars().count(),
                text = %text,
                "ASR 转写完成"
            );
        } else {
            debug!(
                duration = %duration_label,
                latency = %latency_label,
                "ASR 没有识别到文本"
            );
        }

        Ok(TranscriptionResult {
            success,
            text: text.clone(),
            raw_text,
            tokens,
            token_timestamps,
            duration,
            inference_latency: latency,
            confidence: if success { 1.0 } else { 0.0 },
            error: if success {
                None
            } else {
                Some(EMPTY_TRANSCRIPTION_MESSAGE.to_string())
            },
        })
    }

    fn restore_punctuation(&self, text: &str) -> String {
        match self.punctuator.add_punctuation(text) {
            Some(text) => text.trim().to_string(),
            None => {
                warn!("标点恢复失败, 使用原始转写文本");
                text.to_string()
            }
        }
    }
}

fn create_recognizer(files: &AsrModelFiles, hotwords_score: f32) -> Result<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.paraformer = OfflineParaformerModelConfig {
        model: Some(path_string(&files.model)?),
    };
    config.model_config.tokens = Some(path_string(&files.tokens)?);
    config.model_config.num_threads = 2;
    config.model_config.provider = Some("cpu".to_string());
    config.hotwords_score = hotwords_score;

    OfflineRecognizer::create(&config).with_context(|| {
        format!(
            "无法加载 sherpa ASR 模型: model={}, tokens={}",
            files.model.display(),
            files.tokens.display()
        )
    })
}

fn create_punctuator(files: &PuncModelFiles) -> Result<OfflinePunctuation> {
    let mut config = OfflinePunctuationConfig::default();
    config.model.ct_transformer = Some(path_string(&files.model)?);
    config.model.num_threads = 1;
    config.model.provider = Some("cpu".to_string());

    OfflinePunctuation::create(&config).with_context(|| {
        format!(
            "无法加载 sherpa PUNC 模型: model={}",
            files.model.display()
        )
    })
}

pub(crate) fn path_string(path: &Path) -> Result<String> {
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
