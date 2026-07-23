use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::asr_backend::AsrBackend;

mod archive;
mod checksum;
mod discovery;
mod doctor;
mod download;
mod manifest;
mod progress;

#[cfg(test)]
mod tests;

use checksum::sha256_file_without_progress;
use discovery::find_onnx_model_file;

use crate::app::{AppPaths, env_path};

pub use doctor::{loadability_report_for, write_doctor_report_for};

pub const DEFAULT_REVISION: &str = "asr-models";
pub const ASR_MODEL_NAME: &str = "sherpa-onnx-paraformer-zh-2024-03-09";
pub const ASR_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2";
pub const VAD_MODEL_NAME: &str = "silero-vad";
pub const VAD_FILE_NAME: &str = "silero_vad.onnx";
pub const VAD_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
pub const VAD_SHA256: &str = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6";
pub const PUNC_MODEL_NAME: &str =
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8";
pub const PUNC_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2";
pub const IFLYTEK_MODEL_NAME: &str = iflytek_runtime::MODEL_REVISION;
pub const IFLYTEK_MODEL_URL: &str = "https://github.com/azazo1/vocotype-rs/releases/download/models-iflytek-v1.0.0/vocotype-iflytek-model-macos-arm64-v1.0.0.tar.gz";
pub const IFLYTEK_MODEL_SHA256_URL: &str = "https://github.com/azazo1/vocotype-rs/releases/download/models-iflytek-v1.0.0/vocotype-iflytek-model-macos-arm64-v1.0.0.tar.gz.sha256";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Asr,
    Vad,
    Punc,
    Iflytek,
}

impl ModelKind {
    pub fn for_backend(backend: AsrBackend) -> &'static [Self] {
        match backend {
            AsrBackend::Sherpa => &[Self::Asr, Self::Vad, Self::Punc],
            AsrBackend::Iflytek => &[Self::Iflytek],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Vad => "vad",
            Self::Punc => "punc",
            Self::Iflytek => "iflytek",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Asr => ASR_MODEL_NAME,
            Self::Vad => VAD_MODEL_NAME,
            Self::Punc => PUNC_MODEL_NAME,
            Self::Iflytek => IFLYTEK_MODEL_NAME,
        }
    }

    pub fn source_url(self) -> &'static str {
        match self {
            Self::Asr => ASR_MODEL_URL,
            Self::Vad => VAD_MODEL_URL,
            Self::Punc => PUNC_MODEL_URL,
            Self::Iflytek => IFLYTEK_MODEL_URL,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelOptions {
    pub model_dir: Option<PathBuf>,
    pub model_cache_dir: Option<PathBuf>,
    pub revision: String,
}

impl ModelOptions {
    pub fn resolve_paths(&self) -> AppPaths {
        let model_root = self
            .model_dir
            .clone()
            .or_else(|| env_path("VOCOTYPE_MODEL_DIR"));
        let cache_root = self
            .model_cache_dir
            .clone()
            .or_else(|| env_path("VOCOTYPE_MODEL_CACHE_DIR"));
        AppPaths::resolve(model_root, cache_root)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub revision: String,
    pub downloaded_at: DateTime<Utc>,
    pub models: BTreeMap<String, ManifestModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestModel {
    pub source: String,
    pub directory: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct AsrModelFiles {
    pub model: PathBuf,
    pub tokens: PathBuf,
}

#[derive(Clone, Debug)]
pub struct PuncModelFiles {
    pub model: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ModelStore {
    pub paths: AppPaths,
    pub revision: String,
}

impl ModelStore {
    pub fn new(options: &ModelOptions) -> Self {
        Self {
            paths: options.resolve_paths(),
            revision: options.revision.clone(),
        }
    }

    pub fn download_hint(&self) -> String {
        self.download_hint_for(AsrBackend::Sherpa)
    }

    pub fn download_hint_for(&self, backend: AsrBackend) -> String {
        format!(
            "vocotype models download --backend {} --model-dir {}",
            backend,
            self.paths.model_root.display()
        )
    }

    pub fn verify_required_for(&self, backend: AsrBackend) -> Result<()> {
        let missing = self
            .missing_models_for(backend)
            .into_iter()
            .map(|kind| kind.label().to_string())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "模型目录缺少必需文件: {}. 请先运行: {}",
                missing.join(", "),
                self.download_hint_for(backend)
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn missing_models(&self) -> Vec<ModelKind> {
        self.missing_models_for(AsrBackend::Sherpa)
    }

    pub fn missing_models_for(&self, backend: AsrBackend) -> Vec<ModelKind> {
        ModelKind::for_backend(backend)
            .iter()
            .copied()
            .filter(|kind| !self.model_ready(*kind))
            .collect()
    }

    pub fn model_ready(&self, kind: ModelKind) -> bool {
        match kind {
            ModelKind::Asr => self.asr_model_files().is_ok(),
            ModelKind::Vad => self.verify_vad_checksum().is_ok(),
            ModelKind::Punc => self.punc_model_files().is_ok(),
            ModelKind::Iflytek => self.iflytek_model_files().is_ok(),
        }
    }

    pub fn model_dir(&self, kind: ModelKind) -> PathBuf {
        self.paths.model_dir(kind.display_name())
    }

    pub fn asr_model_files(&self) -> Result<AsrModelFiles> {
        let dir = self.model_dir(ModelKind::Asr);
        let model = find_onnx_model_file("ASR", &dir)?;
        let tokens = dir.join("tokens.txt");
        if !tokens.exists() {
            bail!("ASR tokens 文件不存在: {}", tokens.display());
        }
        Ok(AsrModelFiles { model, tokens })
    }

    pub fn vad_model_path(&self) -> Result<PathBuf> {
        let path = self.model_dir(ModelKind::Vad).join(VAD_FILE_NAME);
        if !path.exists() {
            bail!("VAD 模型文件不存在: {}", path.display());
        }
        Ok(path)
    }

    pub fn vad_model_path_for(&self, backend: AsrBackend) -> Result<PathBuf> {
        match backend {
            AsrBackend::Sherpa => self.vad_model_path(),
            AsrBackend::Iflytek => Ok(self.iflytek_model_files()?.vad),
        }
    }

    pub fn punc_model_files(&self) -> Result<PuncModelFiles> {
        let dir = self.model_dir(ModelKind::Punc);
        let model = find_onnx_model_file("PUNC", &dir)?;
        Ok(PuncModelFiles { model })
    }

    pub fn iflytek_model_files(&self) -> Result<iflytek_runtime::EdgeEsrModelFiles> {
        let dir = self.model_dir(ModelKind::Iflytek);
        let files = iflytek_runtime::EdgeEsrModelFiles::from_dir(&dir)?;
        self.verify_iflytek_checksums(&files)?;
        Ok(files)
    }

    pub fn missing_iflytek_files(&self) -> Vec<PathBuf> {
        iflytek_runtime::EdgeEsrModelFiles::missing_from_dir(
            self.model_dir(ModelKind::Iflytek),
        )
    }

    pub fn verify_iflytek_checksums(
        &self,
        files: &iflytek_runtime::EdgeEsrModelFiles,
    ) -> Result<()> {
        let checksum_path = files.root.join("SHA256SUMS");
        let text = std::fs::read_to_string(&checksum_path).with_context(|| {
            format!(
                "讯飞模型缺少 SHA256SUMS: {}",
                checksum_path.display()
            )
        })?;
        let expected = text
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let checksum = fields.next()?;
                let name = fields.next()?.trim_start_matches('*');
                Some((name.to_string(), checksum.to_ascii_lowercase()))
            })
            .collect::<BTreeMap<_, _>>();

        for path in files.required_paths() {
            let name = path
                .strip_prefix(&files.root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let expected_checksum = expected.get(&name).ok_or_else(|| {
                anyhow::anyhow!("讯飞模型 SHA256SUMS 缺少条目: {}", name)
            })?;
            let actual = sha256_file_without_progress(path)?;
            if &actual != expected_checksum {
                bail!(
                    "讯飞模型校验失败: {}, expected {}, got {}",
                    name,
                    expected_checksum,
                    actual
                );
            }
        }
        Ok(())
    }

    pub fn verify_vad_checksum(&self) -> Result<()> {
        let path = self.vad_model_path()?;
        let checksum = sha256_file_without_progress(&path)?;
        if checksum != VAD_SHA256 {
            bail!(
                "VAD 模型校验失败: expected {}, got {}. 请重新运行: {}",
                VAD_SHA256,
                checksum,
                self.download_hint()
            );
        }
        Ok(())
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.paths.manifest_path()
    }

    pub fn read_manifest(&self) -> Result<Option<ModelManifest>> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("无法读取 manifest: {}", path.display()))?;
        let manifest = serde_json::from_str(&text)
            .with_context(|| format!("无法解析 manifest: {}", path.display()))?;
        Ok(Some(manifest))
    }
}
