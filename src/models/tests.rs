use std::path::{Path, PathBuf};

use super::archive::safe_relative_path;
use super::download::vad_temporary_path;
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
    let store = ModelStore::new(&options(
        Some("/tmp/vocotype-models"),
        Some("/tmp/vocotype-cache"),
    ));
    assert_eq!(
        store.paths.model_root,
        PathBuf::from("/tmp/vocotype-models")
    );
    assert_eq!(
        store.paths.model_cache_root,
        PathBuf::from("/tmp/vocotype-cache")
    );
}

#[test]
fn default_missing_models_reports_required_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::new(&ModelOptions {
        model_dir: Some(dir.path().join("models")),
        model_cache_dir: Some(dir.path().join("cache")),
        revision: DEFAULT_REVISION.to_string(),
    });
    let missing = store.missing_models();
    assert_eq!(missing, vec![ModelKind::Iflytek]);
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
    assert!(!store.model_ready(ModelKind::Vad));
}

#[test]
fn vad_ready_requires_valid_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::new(&ModelOptions {
        model_dir: Some(dir.path().join("models")),
        model_cache_dir: Some(dir.path().join("cache")),
        revision: DEFAULT_REVISION.to_string(),
    });
    let vad_dir = store.model_dir(ModelKind::Vad);
    std::fs::create_dir_all(&vad_dir).unwrap();
    std::fs::write(vad_dir.join(VAD_FILE_NAME), []).unwrap();
    assert_eq!(
        store.missing_models_for(AsrBackend::Sherpa),
        vec![ModelKind::Asr, ModelKind::Vad, ModelKind::Punc]
    );
}

#[test]
fn punc_ready_requires_onnx_model() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::new(&ModelOptions {
        model_dir: Some(dir.path().join("models")),
        model_cache_dir: Some(dir.path().join("cache")),
        revision: DEFAULT_REVISION.to_string(),
    });
    let punc_dir = store.model_dir(ModelKind::Punc);
    std::fs::create_dir_all(&punc_dir).unwrap();
    assert!(!store.model_ready(ModelKind::Punc));
    std::fs::write(punc_dir.join("model.int8.onnx"), []).unwrap();
    assert!(store.model_ready(ModelKind::Punc));
}

#[test]
fn vad_temporary_path_does_not_overlap_target_path() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("models");
    let store = ModelStore::new(&ModelOptions {
        model_dir: Some(root.clone()),
        model_cache_dir: Some(root),
        revision: DEFAULT_REVISION.to_string(),
    });
    let target = store.model_dir(ModelKind::Vad).join(VAD_FILE_NAME);
    assert_ne!(vad_temporary_path(&target), target);
    assert_eq!(
        vad_temporary_path(&target).file_name().unwrap(),
        "silero_vad.onnx.download"
    );
}

#[test]
fn unsafe_archive_paths_are_rejected() {
    assert!(safe_relative_path(Path::new("model.onnx")));
    assert!(!safe_relative_path(Path::new("../model.onnx")));
    assert!(!safe_relative_path(Path::new("/tmp/model.onnx")));
}
