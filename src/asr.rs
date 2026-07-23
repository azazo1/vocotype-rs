use std::path::Path;
use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use sherpa_onnx::{
    OfflineParaformerModelConfig, OfflinePunctuation, OfflinePunctuationConfig, OfflineRecognizer,
    OfflineRecognizerConfig,
};
use tracing::{debug, info, warn};

use crate::asr_backend::AsrBackend;
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

#[derive(Clone, Debug)]
pub struct TranscriptionUpdate {
    pub result: TranscriptionResult,
    pub revision: bool,
    pub revision_count: usize,
    pub sequence: usize,
    pub final_result: bool,
}

impl TranscriptionResult {
    pub fn is_empty_transcription(&self) -> bool {
        !self.success
            && self.text.trim().is_empty()
            && self.error.as_deref() == Some(EMPTY_TRANSCRIPTION_MESSAGE)
    }
}

pub struct AsrEngine {
    backend: BackendEngine,
    dictionary: SpeechDictionary,
    english_punctuation: bool,
    strip_trailing_period: bool,
}

enum BackendEngine {
    Sherpa {
        recognizer: OfflineRecognizer,
        punctuator: OfflinePunctuation,
    },
    Iflytek(Arc<iflytek_runtime::EdgeEsrRuntime>),
}

#[derive(Clone, Debug)]
pub struct AsrOptions {
    pub backend: AsrBackend,
    pub dictionary: SpeechDictionary,
    pub hotwords_score: f32,
    pub english_punctuation: bool,
    pub strip_trailing_period: bool,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            backend: AsrBackend::Sherpa,
            dictionary: SpeechDictionary::builtin(),
            hotwords_score: DEFAULT_HOTWORDS_SCORE,
            english_punctuation: false,
            strip_trailing_period: false,
        }
    }
}

impl AsrEngine {
    pub fn load_with_options(store: ModelStore, options: AsrOptions) -> Result<Arc<Self>> {
        store.verify_required_for(options.backend)?;
        let backend = match options.backend {
            AsrBackend::Sherpa => {
                let asr_files = store.asr_model_files()?;
                let punc_files = store.punc_model_files()?;
                let recognizer = create_recognizer(&asr_files, options.hotwords_score)?;
                let punctuator = create_punctuator(&punc_files)?;
                info!(
                    backend = %options.backend,
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
                info!(model = %punc_files.model.display(), "PUNC 模型加载完成");
                BackendEngine::Sherpa {
                    recognizer,
                    punctuator,
                }
            }
            AsrBackend::Iflytek => {
                let files = store.iflytek_model_files()?;
                let runtime = iflytek_runtime::EdgeEsrRuntime::load(
                    files,
                    iflytek_runtime::EdgeEsrRuntimeOptions {
                        postprocess: iflytek_core::PostprocessOptions {
                            english_punctuation: options.english_punctuation,
                            strip_trailing_period: options.strip_trailing_period,
                        },
                        ..iflytek_runtime::EdgeEsrRuntimeOptions::default()
                    },
                )?;
                info!(
                    backend = %options.backend,
                    model = %runtime.files().root.display(),
                    "ASR 模型加载完成"
                );
                BackendEngine::Iflytek(runtime)
            }
        };
        Ok(Arc::new(Self {
            backend,
            dictionary: options.dictionary,
            english_punctuation: options.english_punctuation,
            strip_trailing_period: options.strip_trailing_period,
        }))
    }

    pub fn transcribe_file(&self, audio: &Path) -> Result<TranscriptionResult> {
        let pcm = crate::wav::read_wav_mono_i16(audio, TARGET_SAMPLE_RATE)?;
        self.transcribe_pcm(&pcm)
    }

    pub fn supports_streaming(&self) -> bool {
        matches!(&self.backend, BackendEngine::Iflytek(_))
    }

    pub fn transcribe_pcm(&self, audio: &PcmAudio) -> Result<TranscriptionResult> {
        self.transcribe_pcm_streaming(audio, |_| Ok(()))
    }

    pub fn transcribe_pcm_streaming<Emit>(
        &self,
        audio: &PcmAudio,
        mut emit: Emit,
    ) -> Result<TranscriptionResult>
    where
        Emit: FnMut(TranscriptionUpdate) -> Result<()>,
    {
        let start = Instant::now();
        if audio.samples.is_empty() {
            bail!("音频为空, 跳过转写");
        }

        let samples_i16 = if audio.sample_rate == TARGET_SAMPLE_RATE {
            audio.samples.clone()
        } else {
            crate::wav::resample_linear_i16(&audio.samples, audio.sample_rate, TARGET_SAMPLE_RATE)
        };
        match &self.backend {
            BackendEngine::Sherpa { recognizer, .. } => {
                let samples = i16_to_f32(&samples_i16);
                let stream = recognizer.create_stream();
                stream.accept_waveform(TARGET_SAMPLE_RATE as i32, &samples);
                recognizer.decode(&stream);
                let result = stream
                    .get_result()
                    .ok_or_else(|| anyhow!("ASR 解码没有返回结果"))?;
                let result = self.build_result(
                    audio.duration_seconds(),
                    start.elapsed().as_secs_f32(),
                    DecodedResult {
                        raw_text: result.text.trim().to_string(),
                        tokens: result.tokens,
                        token_timestamps: result.timestamps,
                        confidence: 1.0,
                    },
                    true,
                );
                emit(TranscriptionUpdate {
                    result: result.clone(),
                    revision: false,
                    revision_count: 0,
                    sequence: 1,
                    final_result: true,
                })?;
                Ok(result)
            }
            BackendEngine::Iflytek(runtime) => {
                let mut cursor = 0;
                let mut final_result = None;
                let mut last_text = String::new();
                let mut revision_count = 0;
                runtime.transcribe_streaming_pcm(
                    TARGET_SAMPLE_RATE,
                    |buffer| {
                        if cursor >= samples_i16.len() {
                            return Ok(false);
                        }
                        let end = (cursor + TARGET_SAMPLE_RATE as usize * 64 / 100)
                            .min(samples_i16.len());
                        buffer.extend_from_slice(&samples_i16[cursor..end]);
                        cursor = end;
                        Ok(cursor < samples_i16.len())
                    },
                    |update| {
                        let result = self.build_result(
                            audio.duration_seconds(),
                            start.elapsed().as_secs_f32(),
                            DecodedResult {
                                raw_text: update.transcription.text.trim().to_string(),
                                tokens: update.transcription.tokens,
                                token_timestamps: update.transcription.token_timestamps,
                                confidence: update.transcription.confidence,
                            },
                            update.final_result,
                        );
                        let revision = update.revision
                            || (!last_text.is_empty() && !result.text.starts_with(&last_text));
                        if revision {
                            revision_count += 1;
                        }
                        if !result.text.is_empty() {
                            last_text = result.text.clone();
                        }
                        if update.final_result {
                            final_result = Some(result.clone());
                        }
                        emit(TranscriptionUpdate {
                            result,
                            revision,
                            revision_count,
                            sequence: update.sequence,
                            final_result: update.final_result,
                        })
                    },
                )?;
                final_result.ok_or_else(|| anyhow!("EdgeEsr 流式解码没有返回 final"))
            }
        }
    }

    pub fn transcribe_live_pcm<Read, Emit>(
        &self,
        mut read: Read,
        mut emit: Emit,
    ) -> Result<TranscriptionResult>
    where
        Read: FnMut(&mut Vec<i16>) -> Result<bool>,
        Emit: FnMut(TranscriptionUpdate) -> Result<()>,
    {
        let BackendEngine::Iflytek(runtime) = &self.backend else {
            bail!("当前 ASR 后端不支持实时流式输入");
        };
        let start = Instant::now();
        let input_samples = Cell::new(0usize);
        let mut final_result = None;
        let mut last_text = String::new();
        let mut revision_count = 0;
        runtime.transcribe_streaming_pcm(
            TARGET_SAMPLE_RATE,
            |buffer| {
                let previous_len = buffer.len();
                let has_more = read(buffer)?;
                input_samples.set(input_samples.get() + buffer.len() - previous_len);
                Ok(has_more)
            },
            |update| {
                let duration = input_samples.get() as f32 / TARGET_SAMPLE_RATE as f32;
                let result = self.build_result(
                    duration,
                    start.elapsed().as_secs_f32(),
                    DecodedResult {
                        raw_text: update.transcription.text.trim().to_string(),
                        tokens: update.transcription.tokens,
                        token_timestamps: update.transcription.token_timestamps,
                        confidence: update.transcription.confidence,
                    },
                    update.final_result,
                );
                let revision = update.revision
                    || (!last_text.is_empty() && !result.text.starts_with(&last_text));
                if revision {
                    revision_count += 1;
                }
                if !result.text.is_empty() {
                    last_text = result.text.clone();
                }
                if update.final_result {
                    final_result = Some(result.clone());
                }
                emit(TranscriptionUpdate {
                    result,
                    revision,
                    revision_count,
                    sequence: update.sequence,
                    final_result: update.final_result,
                })
            },
        )?;
        final_result.ok_or_else(|| anyhow!("EdgeEsr 实时流式解码没有返回 final"))
    }

    fn build_result(
        &self,
        duration: f32,
        latency: f32,
        decoded: DecodedResult,
        log_final: bool,
    ) -> TranscriptionResult {
        let raw_text = decoded.raw_text;
        let tokens = decoded.tokens;
        let token_timestamps = decoded.token_timestamps;
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
        if log_final {
            let duration_label = format!("{:.2}", duration);
            let latency_label = format!("{:.2}", latency);
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
        }

        TranscriptionResult {
            success,
            text: text.clone(),
            raw_text,
            tokens,
            token_timestamps,
            duration,
            inference_latency: latency,
            confidence: if success { decoded.confidence } else { 0.0 },
            error: if success {
                None
            } else {
                Some(EMPTY_TRANSCRIPTION_MESSAGE.to_string())
            },
        }
    }

    fn restore_punctuation(&self, text: &str) -> String {
        match &self.backend {
            BackendEngine::Sherpa { punctuator, .. } => {
                match punctuator.add_punctuation(text) {
                    Some(text) => text.trim().to_string(),
                    None => {
                        warn!("标点恢复失败, 使用原始转写文本");
                        text.to_string()
                    }
                }
            }
            BackendEngine::Iflytek(_) => text.to_string(),
        }
    }
}

struct DecodedResult {
    raw_text: String,
    tokens: Vec<String>,
    token_timestamps: Option<Vec<f32>>,
    confidence: f32,
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
