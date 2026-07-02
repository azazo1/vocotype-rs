use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};

use super::archive::extract_tar_bz2;
use super::checksum::sha256_file_without_progress;
use super::progress::{ProgressThrottle, format_bytes};
use super::{
    ASR_MODEL_NAME, ASR_MODEL_URL, ModelKind, ModelManifest, ModelStore, VAD_FILE_NAME,
    VAD_MODEL_URL, VAD_SHA256,
};

impl ModelStore {
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
        let archive = self
            .paths
            .cache_entry_dir(ASR_MODEL_NAME)
            .with_extension("tar.bz2");
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
        let target = target_dir.join(VAD_FILE_NAME);
        let temporary = vad_temporary_path(&target);
        crate::app::ensure_parent(&temporary)?;
        info!(url = VAD_MODEL_URL, path = %temporary.display(), "开始下载 VAD 模型");
        download_to_file(VAD_MODEL_URL, &temporary).await?;
        let checksum = sha256_file_without_progress(&temporary)?;
        if checksum != VAD_SHA256 {
            bail!(
                "VAD 模型校验失败: expected {}, got {}",
                VAD_SHA256,
                checksum
            );
        }

        crate::app::ensure_parent(&target)?;
        std::fs::rename(&temporary, &target)
            .with_context(|| format!("无法保存 VAD 模型到 {}", target.display()))?;
        info!(path = %target.display(), "VAD 模型下载完成");
        Ok(())
    }
}

pub(super) fn vad_temporary_path(target: &Path) -> PathBuf {
    target.with_extension("onnx.download")
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
    let span = tracing::info_span!(
        "下载模型",
        indicatif.pb_show = tracing::field::Empty,
        url = %url,
        path = %path.display(),
    );
    let _enter = span.enter();
    if let Some(total) = total {
        span.pb_set_length(total);
        span.pb_set_style(&download_bar_style());
    } else {
        span.pb_set_style(&download_spinner_style());
    }
    span.pb_set_message(&download_message(path, 0, total));
    span.pb_start();

    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("无法创建下载文件: {}", path.display()))?;
    let mut downloaded = 0_u64;
    let mut progress = ProgressThrottle::new(&span);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("模型下载中断: {}", url))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("无法写入下载文件: {}", path.display()))?;
        downloaded += chunk.len() as u64;
        progress.add(
            chunk.len() as u64,
            Some(&download_message(path, downloaded, total)),
        );
    }
    progress.flush(Some(&download_message(path, downloaded, total)));
    file.flush().await?;
    span.pb_set_finish_message(&format!(
        "下载完成 {} {}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        format_bytes(downloaded)
    ));
    debug!(downloaded, path = %path.display(), "模型下载文件写入完成");
    Ok(())
}

fn download_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} 下载 [{elapsed_precise}] [{bar:32.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

fn download_spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} 下载 [{elapsed_precise}] {bytes} {bytes_per_sec}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

fn download_message(path: &Path, downloaded: u64, total: Option<u64>) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    match total {
        Some(total) => format!(
            "{} {} / {}",
            name,
            format_bytes(downloaded),
            format_bytes(total)
        ),
        None => format!("{} {}", name, format_bytes(downloaded)),
    }
}
