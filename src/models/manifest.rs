use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use tracing::info;

use super::checksum::hash_model_files;
use super::{ManifestModel, ModelKind, ModelManifest, ModelStore};

impl ModelStore {
    pub fn write_manifest(&self) -> Result<ModelManifest> {
        let started = Instant::now();
        info!(path = %self.manifest_path().display(), "开始生成模型 manifest");
        let mut models = BTreeMap::new();
        for kind in ModelKind::all() {
            let dir = self.model_dir(kind);
            let files = hash_model_files(kind, &dir)?;
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
        std::fs::write(&path, text)
            .with_context(|| format!("无法写入 manifest: {}", path.display()))?;
        info!(
            path = %path.display(),
            models = manifest.models.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "模型 manifest 写入完成"
        );
        Ok(manifest)
    }
}
