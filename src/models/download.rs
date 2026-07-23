use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::StatusCode;
use reqwest::header::RANGE;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};

use crate::asr_backend::AsrBackend;

use super::archive::{extract_tar_bz2, extract_tar_gz};
use super::checksum::sha256_file_without_progress;
use super::progress::{ProgressThrottle, format_bytes};
use super::{
    ASR_MODEL_NAME, ASR_MODEL_URL, IFLYTEK_MODEL_NAME, IFLYTEK_MODEL_SHA256_URL,
    IFLYTEK_MODEL_URL, ModelKind, ModelManifest, ModelStore, PUNC_MODEL_NAME, PUNC_MODEL_URL,
    VAD_FILE_NAME, VAD_MODEL_URL, VAD_SHA256,
};

impl ModelStore {
    pub async fn download_backend(&self, backend: AsrBackend) -> Result<ModelManifest> {
        self.paths.ensure_dirs()?;
        for kind in ModelKind::for_backend(backend).iter().copied() {
            self.download_one(kind).await?;
        }
        self.verify_required_for(backend)?;
        self.write_manifest_for(backend)
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
            ModelKind::Punc => self.download_punc(&target_dir).await,
            ModelKind::Iflytek => self.download_iflytek(&target_dir).await,
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

    async fn download_punc(&self, target_dir: &Path) -> Result<()> {
        let archive = self
            .paths
            .cache_entry_dir(PUNC_MODEL_NAME)
            .with_extension("tar.bz2");
        crate::app::ensure_parent(&archive)?;
        info!(url = PUNC_MODEL_URL, path = %archive.display(), "开始下载 PUNC 模型");
        download_to_file(PUNC_MODEL_URL, &archive).await?;
        extract_tar_bz2(&archive, target_dir)?;
        if !self.model_ready(ModelKind::Punc) {
            warn!(path = %target_dir.display(), "PUNC 模型解压后缺少 onnx 模型");
            bail!("PUNC 模型下载不完整");
        }
        info!(path = %target_dir.display(), "PUNC 模型下载完成");
        Ok(())
    }

    async fn download_iflytek(&self, target_dir: &Path) -> Result<()> {
        let archive = self
            .paths
            .cache_entry_dir(IFLYTEK_MODEL_NAME)
            .with_extension("tar.gz");
        let checksum_file = archive.with_extension("tar.gz.sha256");
        crate::app::ensure_parent(&archive)?;
        info!(url = IFLYTEK_MODEL_URL, path = %archive.display(), "开始下载讯飞模型");
        download_to_file(IFLYTEK_MODEL_URL, &archive).await?;
        download_to_file(IFLYTEK_MODEL_SHA256_URL, &checksum_file).await?;

        let expected = read_archive_checksum(&checksum_file)?;
        let actual = sha256_file_without_progress(&archive)?;
        if actual != expected {
            bail!(
                "讯飞模型压缩包校验失败: expected {}, got {}",
                expected,
                actual
            );
        }

        extract_tar_gz(&archive, target_dir)?;
        if !self.model_ready(ModelKind::Iflytek) {
            let missing = self
                .missing_iflytek_files()
                .into_iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>();
            warn!(path = %target_dir.display(), missing = ?missing, "讯飞模型解压后校验失败");
            bail!("讯飞模型下载不完整或校验失败")
        }
        info!(path = %target_dir.display(), "讯飞模型下载完成");
        Ok(())
    }
}

fn read_archive_checksum(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("无法读取模型压缩包校验文件: {}", path.display()))?;
    let checksum = text
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow::anyhow!("模型压缩包校验文件为空: {}", path.display()))?
        .to_ascii_lowercase();
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("模型压缩包 SHA-256 无效: {}", path.display())
    }
    Ok(checksum)
}

pub(super) fn vad_temporary_path(target: &Path) -> PathBuf {
    target.with_extension("onnx.download")
}

async fn download_to_file(url: &str, path: &Path) -> Result<()> {
    const MAX_ATTEMPTS: usize = 5;

    let client = reqwest::Client::new();
    let span = tracing::info_span!(
        "下载模型",
        indicatif.pb_show = tracing::field::Empty,
        url = %url,
        path = %path.display(),
    );
    let _enter = span.enter();
    span.pb_set_style(&download_spinner_style());
    span.pb_start();

    let mut downloaded = tokio::fs::metadata(path)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut progress = ProgressThrottle::new(&span);
    let mut last_error = None;

    for attempt in 1..=MAX_ATTEMPTS {
        let mut request = client.get(url);
        if downloaded > 0 {
            request = request.header(RANGE, format!("bytes={downloaded}-"));
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(anyhow::anyhow!(error));
                if attempt == MAX_ATTEMPTS {
                    break;
                }
                warn!(attempt, downloaded, "模型下载连接失败, 准备重试");
                tokio::time::sleep(download_retry_delay(attempt)).await;
                continue;
            }
        };
        let status = response.status();
        if status == StatusCode::RANGE_NOT_SATISFIABLE && downloaded > 0 {
            tokio::fs::File::create(path)
                .await
                .with_context(|| format!("无法重置下载文件: {}", path.display()))?;
            downloaded = 0;
            warn!(attempt, "服务器拒绝断点位置, 将重新下载模型");
            continue;
        }
        if !status.is_success() {
            bail!("模型下载失败: {} returned {}", url, status);
        }
        if downloaded > 0 && status != StatusCode::PARTIAL_CONTENT {
            downloaded = 0;
            warn!(attempt, "服务器未接受断点续传, 将覆盖现有下载");
        }

        let total = response
            .content_length()
            .map(|remaining| downloaded.saturating_add(remaining));
        if let Some(total) = total {
            span.pb_set_length(total);
            span.pb_set_style(&download_bar_style());
        } else {
            span.pb_set_style(&download_spinner_style());
        }
        span.pb_set_position(downloaded);
        span.pb_set_message(&download_message(path, downloaded, total));

        let mut file = if downloaded == 0 {
            tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .await
        } else {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .await
        }
        .with_context(|| format!("无法打开下载文件: {}", path.display()))?;
        let mut stream = response.bytes_stream();
        let mut interrupted = None;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    interrupted = Some(error);
                    break;
                }
            };
            file.write_all(&chunk)
                .await
                .with_context(|| format!("无法写入下载文件: {}", path.display()))?;
            downloaded += chunk.len() as u64;
            progress.add(
                chunk.len() as u64,
                Some(&download_message(path, downloaded, total)),
            );
        }
        file.flush().await?;

        if let Some(error) = interrupted {
            last_error = Some(anyhow::anyhow!(error));
            if attempt == MAX_ATTEMPTS {
                break;
            }
            warn!(attempt, downloaded, "模型下载中断, 将从当前进度继续");
            tokio::time::sleep(download_retry_delay(attempt)).await;
            continue;
        }
        if total.is_some_and(|total| downloaded != total) {
            last_error = Some(anyhow::anyhow!(
                "模型下载长度不完整: expected {}, got {}",
                total.unwrap_or_default(),
                downloaded
            ));
            if attempt == MAX_ATTEMPTS {
                break;
            }
            warn!(attempt, downloaded, "模型下载长度不完整, 将从当前进度继续");
            tokio::time::sleep(download_retry_delay(attempt)).await;
            continue;
        }

        progress.flush(Some(&download_message(path, downloaded, total)));
        span.pb_set_finish_message(&format!(
            "下载完成 {} {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model"),
            format_bytes(downloaded)
        ));
        debug!(downloaded, path = %path.display(), "模型下载文件写入完成");
        return Ok(());
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("模型下载重试次数已耗尽")))
        .with_context(|| format!("模型下载中断: {}", url))
}

fn download_retry_delay(attempt: usize) -> std::time::Duration {
    std::time::Duration::from_secs(1_u64 << attempt.saturating_sub(1).min(3))
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
