use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::Result;
use sha2::{Digest, Sha256};
use tracing::info;
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};

use super::progress::{ProgressThrottle, format_bytes};
use super::ModelKind;

pub(super) fn hash_model_files(kind: ModelKind, dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut files = BTreeMap::new();
    if !dir.exists() {
        info!(path = %dir.display(), "模型目录不存在, 跳过校验和计算");
        return Ok(files);
    }

    let mut entries = Vec::new();
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        let relative = path
            .strip_prefix(dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        entries.push((relative, path, size));
    }

    let total_bytes = entries
        .iter()
        .map(|(_, _, size)| *size)
        .fold(0_u64, u64::saturating_add);
    let span = checksum_span(kind, dir, total_bytes);
    let _enter = span.enter();

    for (relative, path, _size) in entries {
        let digest = sha256_file(&path, &span)?;
        files.insert(relative, digest);
    }

    span.pb_set_finish_message(&format!(
        "校验完成 {} {}",
        kind.label(),
        format_bytes(total_bytes)
    ));
    Ok(files)
}

fn checksum_span(kind: ModelKind, dir: &Path, total_bytes: u64) -> tracing::Span {
    let span = tracing::info_span!(
        "计算校验和",
        indicatif.pb_show = tracing::field::Empty,
        model = kind.label(),
        path = %dir.display(),
    );
    span.pb_set_length(total_bytes);
    span.pb_set_style(&checksum_bar_style());
    span.pb_set_message(&format!("{} {}", kind.label(), format_bytes(total_bytes)));
    span.pb_start();
    span
}

fn checksum_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} sha256 [{elapsed_precise}] [{bar:32.magenta/blue}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

fn sha256_file(path: &Path, span: &tracing::Span) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut progress = ProgressThrottle::new(span);
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        progress.add(read as u64, None);
    }
    progress.flush(None);
    let digest = hasher.finalize();
    Ok(digest
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<String>())
}

pub(super) fn sha256_file_without_progress(path: &Path) -> Result<String> {
    let span = tracing::Span::none();
    sha256_file(path, &span)
}
