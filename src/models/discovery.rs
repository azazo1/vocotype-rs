use std::path::{Path, PathBuf};

use anyhow::Result;

pub(super) fn find_onnx_model_file(kind: &str, dir: &Path) -> Result<PathBuf> {
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
            if path
                .extension()
                .is_some_and(|extension| extension == "onnx")
            {
                candidates.push(path.to_path_buf());
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("{} 模型文件不存在: {}", kind, dir.display()))
}
