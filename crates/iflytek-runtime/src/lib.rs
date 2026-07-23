use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use iflytek_core::{
    EdgeEsrPostprocessor, MemoryAttentionConfig, MemoryTryAttention, PostprocessOptions,
    SAMPLE_RATE,
};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use ort::session::Session;
use tracing::info;

mod decoder;
mod encoder;
mod recognizer;
mod tensor;
mod vad;

pub use vad::{EdgeEsrVad, EdgeEsrVadConfig, EdgeEsrVadSegment, EdgeEsrVadSegmentReason};

pub const MODEL_REVISION: &str = "iflytek-edgeesr-v1.0.0";
pub const MODEL_RELEASE_TAG: &str = "models-iflytek-v1.0.0";
pub const MODEL_RELEASE_ASSET: &str =
    "vocotype-iflytek-model-macos-arm64-v1.0.0.tar.gz";
pub const MODEL_RELEASE_SHA256_ASSET: &str =
    "vocotype-iflytek-model-macos-arm64-v1.0.0.tar.gz.sha256";

pub const REQUIRED_MODEL_FILES: [&str; 14] = [
    "vgg_encoder.onnx",
    "conformer_encoder.onnx",
    "decoder_part1.onnx",
    "decoder_part2.onnx",
    "vad.onnx",
    "attention_weights.npz",
    "tokens.txt",
    "number-normalization.npz",
    "number-vocabulary.txt",
    "number-not-change.txt.gz",
    "punctuation-bert.npz",
    "punctuation-vocabulary.txt",
    "punctuation-maplist.bin",
    "replacements.txt.gz",
];

#[derive(Clone, Debug)]
pub struct EdgeEsrModelFiles {
    pub root: PathBuf,
    pub vgg_encoder: PathBuf,
    pub conformer_encoder: PathBuf,
    pub decoder_part1: PathBuf,
    pub decoder_part2: PathBuf,
    pub vad: PathBuf,
    pub attention_weights: PathBuf,
    pub tokens: PathBuf,
    pub number_normalization: PathBuf,
    pub number_vocabulary: PathBuf,
    pub number_not_change: PathBuf,
    pub punctuation_bert: PathBuf,
    pub punctuation_vocabulary: PathBuf,
    pub punctuation_maplist: PathBuf,
    pub replacements: PathBuf,
}

impl EdgeEsrModelFiles {
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let path = |name: &str| root.join(name);
        let files = Self {
            root: root.clone(),
            vgg_encoder: path("vgg_encoder.onnx"),
            conformer_encoder: path("conformer_encoder.onnx"),
            decoder_part1: path("decoder_part1.onnx"),
            decoder_part2: path("decoder_part2.onnx"),
            vad: path("vad.onnx"),
            attention_weights: path("attention_weights.npz"),
            tokens: path("tokens.txt"),
            number_normalization: path("number-normalization.npz"),
            number_vocabulary: path("number-vocabulary.txt"),
            number_not_change: path("number-not-change.txt.gz"),
            punctuation_bert: path("punctuation-bert.npz"),
            punctuation_vocabulary: path("punctuation-vocabulary.txt"),
            punctuation_maplist: path("punctuation-maplist.bin"),
            replacements: path("replacements.txt.gz"),
        };
        for file in files.required_paths() {
            if !file.exists() {
                bail!("missing EdgeEsr model file: {}", file.display())
            }
        }
        Ok(files)
    }

    pub fn missing_from_dir(root: impl AsRef<Path>) -> Vec<PathBuf> {
        let root = root.as_ref();
        REQUIRED_MODEL_FILES
            .iter()
            .map(|name| root.join(name))
            .filter(|path| !path.is_file())
            .collect()
    }

    pub fn required_paths(&self) -> [&Path; 14] {
        [
            &self.vgg_encoder,
            &self.conformer_encoder,
            &self.decoder_part1,
            &self.decoder_part2,
            &self.vad,
            &self.attention_weights,
            &self.tokens,
            &self.number_normalization,
            &self.number_vocabulary,
            &self.number_not_change,
            &self.punctuation_bert,
            &self.punctuation_vocabulary,
            &self.punctuation_maplist,
            &self.replacements,
        ]
    }
}

#[derive(Clone, Debug, Default)]
pub struct EdgeEsrTranscription {
    pub text: String,
    pub tokens: Vec<String>,
    pub token_timestamps: Option<Vec<f32>>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default)]
pub struct EdgeEsrOptions {
    pub postprocess: PostprocessOptions,
}

struct EdgeEsrSessions {
    vgg_encoder: Mutex<Session>,
    conformer_encoder: Mutex<Session>,
    decoder_part1: Mutex<Session>,
    decoder_part2: Mutex<Session>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DecoderActiveRows;

impl DecoderActiveRows {
    fn load() -> Result<Self> {
        Ok(Self)
    }

    pub(crate) fn set(&self, rows: i32) -> Result<()> {
        iflytek_core::set_decoder_active_rows(rows)
    }
}

#[derive(Clone)]
pub struct EdgeEsrRuntime {
    files: EdgeEsrModelFiles,
    postprocessor: EdgeEsrPostprocessor,
    sessions: Arc<EdgeEsrSessions>,
    active_rows: DecoderActiveRows,
    decoder_lock: Arc<Mutex<()>>,
    attention_weights: Arc<(Vec<f32>, Vec<f32>)>,
    vocabulary: Arc<Vec<String>>,
}

impl EdgeEsrRuntime {
    pub fn load(files: EdgeEsrModelFiles, options: EdgeEsrOptions) -> Result<Arc<Self>> {
        let started = std::time::Instant::now();
        init_onnx_runtime_api()?;
        let active_rows = DecoderActiveRows::load()?;
        let postprocessor = EdgeEsrPostprocessor::load(
            &files.number_normalization,
            &files.number_vocabulary,
            &files.number_not_change,
            &files.punctuation_bert,
            &files.punctuation_vocabulary,
            &files.punctuation_maplist,
            &files.replacements,
            options.postprocess,
        )?;
        let attention_weights = Arc::new(load_attention_weights(&files.attention_weights)?);
        let vocabulary = Arc::new(load_vocabulary(&files.tokens)?);
        info!(path = %files.root.display(), "loading EdgeEsr model files");
        let sessions = Arc::new(EdgeEsrSessions {
            vgg_encoder: Mutex::new(load_session(&files.vgg_encoder)?),
            conformer_encoder: Mutex::new(load_session(&files.conformer_encoder)?),
            decoder_part1: Mutex::new(load_session(&files.decoder_part1)?),
            decoder_part2: Mutex::new(load_session(&files.decoder_part2)?),
        });
        let runtime = Arc::new(Self {
            files,
            postprocessor,
            sessions,
            active_rows,
            decoder_lock: Arc::new(Mutex::new(())),
            attention_weights,
            vocabulary,
        });
        info!(elapsed_ms = started.elapsed().as_millis(), "EdgeEsr model files validated");
        Ok(runtime)
    }

    pub fn files(&self) -> &EdgeEsrModelFiles {
        &self.files
    }

    pub fn postprocessor(&self) -> &EdgeEsrPostprocessor {
        &self.postprocessor
    }

    pub fn decoder_guard(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        self.decoder_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("EdgeEsr decoder lock poisoned"))
    }

    pub fn session_input_names(&self) -> Result<Vec<(&'static str, Vec<String>)>> {
        Ok(vec![
            ("vgg_encoder", session_inputs(&self.sessions.vgg_encoder)?),
            (
                "conformer_encoder",
                session_inputs(&self.sessions.conformer_encoder)?,
            ),
            (
                "decoder_part1",
                session_inputs(&self.sessions.decoder_part1)?,
            ),
            (
                "decoder_part2",
                session_inputs(&self.sessions.decoder_part2)?,
            ),
        ])
    }

    pub fn transcribe_pcm(
        &self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<EdgeEsrTranscription> {
        if sample_rate as usize != SAMPLE_RATE {
            bail!("EdgeEsr ASR requires a 16000 Hz sample rate")
        }
        if samples.is_empty() {
            bail!("EdgeEsr ASR input is empty")
        }
        let started = std::time::Instant::now();
        let _decoder_guard = self.decoder_guard()?;
        let mut vgg = lock_session(&self.sessions.vgg_encoder)?;
        let mut conformer = lock_session(&self.sessions.conformer_encoder)?;
        let mut decoder1 = lock_session(&self.sessions.decoder_part1)?;
        let mut decoder2 = lock_session(&self.sessions.decoder_part2)?;
        let encoder = encoder::EdgeEsrEncoder::new(&mut vgg, &mut conformer)?;
        let attention = MemoryTryAttention::new(
            self.attention_weights.0.clone(),
            self.attention_weights.1.clone(),
            MemoryAttentionConfig::default(),
        )?;
        let decoder = decoder::EdgeEsrDecoder::new(
            &mut decoder1,
            &mut decoder2,
            attention,
            self.active_rows,
        )?;
        let mut recognizer = recognizer::EdgeEsrRecognizer::new(encoder, decoder);
        let startup_chunks = recognizer.prime_encoder()?;
        recognizer.accept_pcm(samples, false)?;
        recognizer.accept_pcm(&vec![0; SAMPLE_RATE * 4 / 5], true)?;
        let (path, normalized_score) = recognizer.best_path().unwrap_or((&[], -20_000.0));
        let mut raw_tokens = Vec::new();
        for token in path {
            let token_id = *token - 1;
            if token_id < 0 {
                continue;
            }
            let token = self
                .vocabulary
                .get(token_id as usize)
                .cloned()
                .unwrap_or_else(|| format!("<score-id:{}>", token_id));
            if !matches!(
                token.as_str(),
                "<s>" | "</s>" | "<unk>" | "<NOISE>" | "<CAT>"
            ) {
                raw_tokens.push(token);
            }
        }
        let tokens = merge_bpe_continuations(raw_tokens);
        let processed = self.postprocessor.process(&tokens, true)?;
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            startup_chunks,
            token_count = processed.tokens.len(),
            "EdgeEsr transcription completed"
        );
        Ok(EdgeEsrTranscription {
            text: processed.text,
            tokens: processed.tokens,
            token_timestamps: None,
            confidence: normalized_score.exp().clamp(0.0, 1.0),
        })
    }
}

fn merge_bpe_continuations(tokens: Vec<String>) -> Vec<String> {
    let mut merged: Vec<String> = Vec::new();
    for token in tokens {
        if let Some(continuation) = token.strip_prefix("@@") {
            if let Some(previous) = merged.last_mut() {
                previous.push_str(continuation);
            } else if !continuation.is_empty() {
                merged.push(continuation.to_string());
            }
        } else {
            merged.push(token);
        }
    }
    merged
}

fn lock_session(session: &Mutex<Session>) -> Result<std::sync::MutexGuard<'_, Session>> {
    session
        .lock()
        .map_err(|_| anyhow::anyhow!("EdgeEsr ONNX session lock poisoned"))
}

fn load_attention_weights(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let stream = File::open(path).map_err(|error| {
        anyhow::anyhow!(
            "unable to open attention weights {}: {}",
            path.display(),
            error
        )
    })?;
    let mut archive = NpzReader::new(BufReader::new(stream))?;
    let weight: ArrayD<f32> = archive
        .by_name("weight_at_v.npy")
        .map_err(|error| anyhow::anyhow!("attention archive is missing weight_at_v: {}", error))?;
    let weight_stop: ArrayD<f32> = archive
        .by_name("weight_at_v_stop.npy")
        .map_err(|error| {
            anyhow::anyhow!("attention archive is missing weight_at_v_stop: {}", error)
        })?;
    let weight = weight.into_raw_vec_and_offset();
    let weight_stop = weight_stop.into_raw_vec_and_offset();
    if weight.1.unwrap_or(0) != 0 || weight_stop.1.unwrap_or(0) != 0 {
        bail!("attention archive arrays must be contiguous")
    }
    Ok((weight.0, weight_stop.0))
}

fn load_vocabulary(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|error| {
        anyhow::anyhow!("unable to read EdgeEsr vocabulary {}: {}", path.display(), error)
    })?;
    let vocabulary = content.lines().map(str::to_string).collect::<Vec<_>>();
    if vocabulary.is_empty() {
        bail!("EdgeEsr vocabulary is empty")
    }
    Ok(vocabulary)
}

fn load_session(path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(|error| anyhow::anyhow!("unable to create ONNX session builder: {}", error))?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!("unable to configure ONNX session: {}", error))?
        .with_operators(iflytek_ops::operator_domain().map_err(|error| {
            anyhow::anyhow!(
                "unable to create {} custom-op domain: {}",
                iflytek_core::CUSTOM_OP_DOMAIN,
                error
            )
        })?)
        .map_err(|error| anyhow::anyhow!("unable to register custom-op domain: {}", error))?
        .commit_from_file(path)
        .map_err(|error| anyhow::anyhow!("unable to load ONNX model {}: {}", path.display(), error))
}

fn load_plain_session(path: &Path) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|error| anyhow::anyhow!("unable to create ONNX session builder: {}", error))?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!("unable to configure ONNX session: {}", error))?;
    builder
        .commit_from_file(path)
        .map_err(|error| anyhow::anyhow!("unable to load ONNX model {}: {}", path.display(), error))
}

fn session_inputs(session: &Mutex<Session>) -> Result<Vec<String>> {
    let session = session
        .lock()
        .map_err(|_| anyhow::anyhow!("EdgeEsr ONNX session lock poisoned"))?;
    Ok(session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect())
}

pub fn init_onnx_runtime_api() -> Result<()> {
    let base = unsafe { ort_sys::OrtGetApiBase() };
    if base.is_null() {
        bail!("linked ONNX Runtime did not expose OrtGetApiBase")
    }
    let api = unsafe { ((*base).GetApi)(ort_sys::ORT_API_VERSION) };
    if api.is_null() {
        bail!(
            "linked ONNX Runtime does not provide API version {}",
            ort_sys::ORT_API_VERSION
        )
    }
    let _ = ort::set_api(unsafe { (*api).clone() });
    Ok(())
}

pub fn model_dir(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(MODEL_REVISION)
}

pub fn model_load_error(root: &Path, error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!(
        "unable to load EdgeEsr model from {}: {}. Download release {}",
        root.display(),
        error,
        MODEL_RELEASE_TAG
    )
}

#[cfg(test)]
mod tests {
    use super::merge_bpe_continuations;

    #[test]
    fn bpe_continuations_merge_into_the_previous_token() {
        let tokens = vec!["co", "@@d", "@@ex", "的", "a", "@@i"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(merge_bpe_continuations(tokens), ["codex", "的", "ai"]);
    }

    #[test]
    fn orphan_bpe_continuation_does_not_leak_marker() {
        assert_eq!(
            merge_bpe_continuations(vec!["@@edge".to_string()]),
            ["edge"]
        );
    }
}
