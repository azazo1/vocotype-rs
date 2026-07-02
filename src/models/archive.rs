use std::fs::File;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use tracing::debug;
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};

use super::progress::{ProgressRead, format_bytes};

pub(super) fn extract_tar_bz2(archive: &Path, target_dir: &Path) -> Result<()> {
    let file = File::open(archive)
        .with_context(|| format!("无法打开模型压缩包: {}", archive.display()))?;
    let total_bytes = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let span = extract_span(archive, total_bytes);
    let _enter = span.enter();
    let mut reader = ProgressRead::new(file, &span);

    {
        let decoder = BzDecoder::new(&mut reader);
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
    }

    reader.flush();
    span.pb_set_finish_message(&format!(
        "解压完成 {} {}",
        archive
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        format_bytes(total_bytes)
    ));
    Ok(())
}

fn extract_span(archive: &Path, total_bytes: u64) -> tracing::Span {
    let span = tracing::info_span!(
        "解压模型",
        indicatif.pb_show = tracing::field::Empty,
        path = %archive.display(),
    );
    span.pb_set_length(total_bytes);
    span.pb_set_style(&extract_bar_style());
    span.pb_start();
    span
}

fn extract_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} 解压 [{elapsed_precise}] [{bar:32.yellow/blue}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

fn strip_archive_root(path: &Path) -> PathBuf {
    let mut components = path.components();
    let _ = components.next();
    components.as_path().to_path_buf()
}

pub(super) fn safe_relative_path(path: &Path) -> bool {
    path.components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}
