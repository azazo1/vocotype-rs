use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
use discovery::find_asr_model_file;

use crate::app::{AppPaths, env_path};

pub use doctor::{loadability_report, write_doctor_report};

pub const DEFAULT_REVISION: &str = "asr-models";
pub const ASR_MODEL_NAME: &str = "sherpa-onnx-paraformer-zh-2024-03-09";
pub const ASR_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2";
pub const VAD_MODEL_NAME: &str = "silero-vad";
pub const VAD_FILE_NAME: &str = "silero_vad.onnx";
pub const VAD_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
pub const VAD_SHA256: &str = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Asr,
    Vad,
}

impl ModelKind {
    pub fn all() -> [Self; 2] {
        [Self::Asr, Self::Vad]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Vad => "vad",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Asr => ASR_MODEL_NAME,
            Self::Vad => VAD_MODEL_NAME,
        }
    }

    pub fn source_url(self) -> &'static str {
        match self {
            Self::Asr => ASR_MODEL_URL,
            Self::Vad => VAD_MODEL_URL,
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
        format!(
            "vocotype models download --model-dir {}",
            self.paths.model_root.display()
        )
    }

    pub fn verify_required(&self) -> Result<()> {
        let missing = self
            .missing_models()
            .into_iter()
            .map(|kind| kind.label().to_string())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "模型目录缺少必需文件: {}. 请先运行: {}",
                missing.join(", "),
                self.download_hint()
            );
        }
        Ok(())
    }

    pub fn missing_models(&self) -> Vec<ModelKind> {
        ModelKind::all()
            .into_iter()
            .filter(|kind| !self.model_ready(*kind))
            .collect()
    }

    pub fn model_ready(&self, kind: ModelKind) -> bool {
        match kind {
            ModelKind::Asr => self.asr_model_files().is_ok(),
            ModelKind::Vad => self.verify_vad_checksum().is_ok(),
        }
    }

    pub fn model_dir(&self, kind: ModelKind) -> PathBuf {
        self.paths.model_dir(kind.display_name())
    }

    pub fn asr_model_files(&self) -> Result<AsrModelFiles> {
        let dir = self.model_dir(ModelKind::Asr);
        let model = find_asr_model_file(&dir)?;
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
