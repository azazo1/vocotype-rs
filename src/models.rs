use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use bzip2::read::BzDecoder;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use crate::app::{AppPaths, env_path};

pub const DEFAULT_REVISION: &str = "asr-models";
pub const ASR_MODEL_NAME: &str = "sherpa-onnx-paraformer-zh-2024-03-09";
pub const ASR_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2";
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
            ModelKind::Vad => self.vad_model_path().map(|path| path.exists()).unwrap_or(false),
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

    pub fn write_manifest(&self) -> Result<ModelManifest> {
        let mut models = BTreeMap::new();
        for kind in ModelKind::all() {
            let dir = self.model_dir(kind);
            let files = hash_model_files(&dir)?;
            models.insert(
                kind.label().to_string(),
                ManifestModel {
                    source: kind.source_url().to_string(),
                    directory: kind.display_name().to_string(),
                    files,
                },
            );
        }

        let manifest = ModelManifest {
            revision: self.revision.clone(),
            downloaded_at: Utc::now(),
            models,
        };
        let path = self.manifest_path();
        crate::app::ensure_parent(&path)?;
        let text = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&path, text).with_context(|| format!("无法写入 manifest: {}", path.display()))?;
        Ok(manifest)
    }

    pub async fn download_all(&self) -> Result<ModelManifest> {
        self.paths.ensure_dirs()?;
        for kind in ModelKind::all() {
            self.download_one(kind).await?;
        }
        self.verify_required()?;
        self.write_manifest()
    }

    async fn download_one(&self, kind: ModelKind) -> Result<()> {
        let target_dir = self.model_dir(kind);
        if self.model_ready(kind) {
            info!(model = kind.label(), path = %target_dir.display(), "模型已存在, 跳过下载");
            return Ok(());
        }

        std::fs::create_dir_all(&target_dir)?;
        match kind {
            ModelKind::Asr => self.download_asr(&target_dir).await,
            ModelKind::Vad => self.download_vad(&target_dir).await,
        }
    }

    async fn download_asr(&self, target_dir: &Path) -> Result<()> {
        let archive = self.paths.cache_entry_dir(ASR_MODEL_NAME).with_extension("tar.bz2");
        crate::app::ensure_parent(&archive)?;
        info!(url = ASR_MODEL_URL, path = %archive.display(), "开始下载 ASR 模型");
        download_to_file(ASR_MODEL_URL, &archive).await?;
        extract_tar_bz2(&archive, target_dir)?;
        if !self.model_ready(ModelKind::Asr) {
            warn!(path = %target_dir.display(), "ASR 模型解压后缺少 onnx 模型或 tokens.txt");
            bail!("ASR 模型下载不完整");
        }
        info!(path = %target_dir.display(), "ASR 模型下载完成");
        Ok(())
    }

    async fn download_vad(&self, target_dir: &Path) -> Result<()> {
        let archive = self.paths.cache_entry_dir(VAD_MODEL_NAME).join(VAD_FILE_NAME);
        crate::app::ensure_parent(&archive)?;
        info!(url = VAD_MODEL_URL, path = %archive.display(), "开始下载 VAD 模型");
        download_to_file(VAD_MODEL_URL, &archive).await?;
        let checksum = sha256_file(&archive)?;
        if checksum != VAD_SHA256 {
            bail!("VAD 模型校验失败: expected {}, got {}", VAD_SHA256, checksum);
        }

        let target = target_dir.join(VAD_FILE_NAME);
        crate::app::ensure_parent(&target)?;
        std::fs::copy(&archive, &target)
            .with_context(|| format!("无法复制 VAD 模型到 {}", target.display()))?;
        info!(path = %target.display(), "VAD 模型下载完成");
        Ok(())
    }
}

async fn download_to_file(url: &str, path: &Path) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("模型下载请求失败: {}", url))?;
    if !response.status().is_success() {
        bail!("模型下载失败: {} returned {}", url, response.status());
    }

    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("无法创建下载文件: {}", path.display()))?;
    let mut downloaded = 0_u64;
    let mut last_logged = 0_u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("模型下载中断: {}", url))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("无法写入下载文件: {}", path.display()))?;
        downloaded += chunk.len() as u64;
        if downloaded.saturating_sub(last_logged) >= 64 * 1024 * 1024 {
            last_logged = downloaded;
            match total {
                Some(total) => info!(downloaded, total, "模型下载进度"),
                None => info!(downloaded, "模型下载进度"),
            }
        }
    }
    file.flush().await?;
    debug!(downloaded, path = %path.display(), "模型下载文件写入完成");
    Ok(())
}

fn extract_tar_bz2(archive: &Path, target_dir: &Path) -> Result<()> {
    let file = File::open(archive)
        .with_context(|| format!("无法打开模型压缩包: {}", archive.display()))?;
    let decoder = BzDecoder::new(file);
    let mut tar = tar::Archive::new(decoder);

    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let stripped = strip_archive_root(&path);
        if stripped.as_os_str().is_empty() || !safe_relative_path(&stripped) {
            continue;
        }

        let output = target_dir.join(stripped);
        let entry_type = entry.header().entry_type();
        if entry_type.is_dir() {
            std::fs::create_dir_all(&output)
                .with_context(|| format!("无法创建模型目录: {}", output.display()))?;
        } else if entry_type.is_file() {
            crate::app::ensure_parent(&output)?;
            entry
                .unpack(&output)
                .with_context(|| format!("无法解压模型文件: {}", output.display()))?;
        } else {
            debug!(path = %output.display(), "跳过非普通模型文件");
        }
    }

    Ok(())
}

fn strip_archive_root(path: &Path) -> PathBuf {
    let mut components = path.components();
    let _ = components.next();
    components.as_path().to_path_buf()
}

fn safe_relative_path(path: &Path) -> bool {
    path.components().all(|component| {
        matches!(
            component,
            Component::Normal(_) | Component::CurDir
        )
    })
}

fn find_asr_model_file(dir: &Path) -> Result<PathBuf> {
    let preferred = [dir.join("model.onnx"), dir.join("model.int8.onnx")];
    for path in preferred {
        if path.exists() {
            return Ok(path);
        }
    }

    let mut candidates = Vec::new();
    if dir.exists() {
        for entry in walkdir::WalkDir::new(dir).max_depth(2) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().is_some_and(|extension| extension == "onnx") {
                candidates.push(path.to_path_buf());
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("ASR 模型文件不存在: {}", dir.display()))
}

fn hash_model_files(dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut files = BTreeMap::new();
    if !dir.exists() {
        return Ok(files);
    }

    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let relative = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.insert(relative, sha256_file(path)?);
    }

    Ok(files)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn write_doctor_report(store: &ModelStore, mut writer: impl Write) -> Result<()> {
    writeln!(writer, "model_dir={}", store.paths.model_root.display())?;
    writeln!(writer, "model_cache_dir={}", store.paths.model_cache_root.display())?;
    writeln!(writer, "revision={}", store.revision)?;
    for kind in ModelKind::all() {
        writeln!(
            writer,
            "{}={}",
            kind.label(),
            if store.model_ready(kind) { "ready" } else { "missing" }
        )?;
    }
    match store.read_manifest()? {
        Some(manifest) => writeln!(writer, "manifest=ready {}", manifest.downloaded_at)?,
        None => writeln!(writer, "manifest=missing")?,
    }
    if !store.missing_models().is_empty() {
        writeln!(writer, "hint={}", store.download_hint())?;
    }
    Ok(())
}

pub fn loadability_report(store: &ModelStore, mut writer: impl Write) -> Result<()> {
    store.verify_required()?;
    let _engine = crate::asr::AsrEngine::load(store.clone())?;
    let vad_model = store.vad_model_path()?;
    let _vad = crate::vad::VadSegmenter::new(crate::vad::VadConfig::default(), &vad_model)?;
    writeln!(writer, "sherpa_asr=loadable")?;
    writeln!(writer, "sherpa_vad=loadable")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(model_dir: Option<&str>, cache_dir: Option<&str>) -> ModelOptions {
        ModelOptions {
            model_dir: model_dir.map(PathBuf::from),
            model_cache_dir: cache_dir.map(PathBuf::from),
            revision: DEFAULT_REVISION.to_string(),
        }
    }

    #[test]
    fn cli_paths_override_defaults() {
        let store = ModelStore::new(&options(Some("/tmp/vocotype-models"), Some("/tmp/vocotype-cache")));
        assert_eq!(store.paths.model_root, PathBuf::from("/tmp/vocotype-models"));
        assert_eq!(store.paths.model_cache_root, PathBuf::from("/tmp/vocotype-cache"));
    }

    #[test]
    fn missing_models_reports_required_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(&ModelOptions {
            model_dir: Some(dir.path().join("models")),
            model_cache_dir: Some(dir.path().join("cache")),
            revision: DEFAULT_REVISION.to_string(),
        });
        let missing = store.missing_models();
        assert_eq!(missing, vec![ModelKind::Asr, ModelKind::Vad]);
    }

    #[test]
    fn asr_ready_requires_onnx_and_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(&ModelOptions {
            model_dir: Some(dir.path().join("models")),
            model_cache_dir: Some(dir.path().join("cache")),
            revision: DEFAULT_REVISION.to_string(),
        });
        let asr_dir = store.model_dir(ModelKind::Asr);
        std::fs::create_dir_all(&asr_dir).unwrap();
        std::fs::write(asr_dir.join("model.onnx"), []).unwrap();
        assert!(!store.model_ready(ModelKind::Asr));
        std::fs::write(asr_dir.join("tokens.txt"), []).unwrap();
        assert!(store.model_ready(ModelKind::Asr));
    }

    #[test]
    fn asr_model_discovery_accepts_int8_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(&ModelOptions {
            model_dir: Some(dir.path().join("models")),
            model_cache_dir: Some(dir.path().join("cache")),
            revision: DEFAULT_REVISION.to_string(),
        });
        let asr_dir = store.model_dir(ModelKind::Asr);
        std::fs::create_dir_all(&asr_dir).unwrap();
        std::fs::write(asr_dir.join("model.int8.onnx"), []).unwrap();
        std::fs::write(asr_dir.join("tokens.txt"), []).unwrap();
        let files = store.asr_model_files().unwrap();
        assert_eq!(files.model, asr_dir.join("model.int8.onnx"));
    }

    #[test]
    fn vad_ready_requires_silero_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(&ModelOptions {
            model_dir: Some(dir.path().join("models")),
            model_cache_dir: Some(dir.path().join("cache")),
            revision: DEFAULT_REVISION.to_string(),
        });
        let vad_dir = store.model_dir(ModelKind::Vad);
        std::fs::create_dir_all(&vad_dir).unwrap();
        std::fs::write(vad_dir.join(VAD_FILE_NAME), []).unwrap();
        assert!(store.model_ready(ModelKind::Vad));
    }

    #[test]
    fn unsafe_archive_paths_are_rejected() {
        assert!(safe_relative_path(Path::new("model.onnx")));
        assert!(!safe_relative_path(Path::new("../model.onnx")));
        assert!(!safe_relative_path(Path::new("/tmp/model.onnx")));
    }
}
