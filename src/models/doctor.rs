use std::io::Write;

use anyhow::Result;
use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};

use super::{ModelKind, ModelStore};

pub fn write_doctor_report(store: &ModelStore, mut writer: impl Write) -> Result<()> {
    writeln!(writer, "model_dir={}", store.paths.model_root.display())?;
    writeln!(
        writer,
        "model_cache_dir={}",
        store.paths.model_cache_root.display()
    )?;
    writeln!(writer, "revision={}", store.revision)?;
    for kind in ModelKind::all() {
        writeln!(
            writer,
            "{}={}",
            kind.label(),
            if store.model_ready(kind) {
                "ready"
            } else {
                "missing"
            }
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
    store.verify_vad_checksum()?;
    let vad_model = store.vad_model_path()?;
    let _vad = crate::vad::VadSegmenter::new(crate::vad::VadConfig::default(), &vad_model)?;
    let punc_files = store.punc_model_files()?;
    let mut punc_config = OfflinePunctuationConfig::default();
    punc_config.model.ct_transformer = Some(crate::asr::path_string(&punc_files.model)?);
    let _punc = OfflinePunctuation::create(&punc_config)
        .ok_or_else(|| anyhow::anyhow!("无法加载 sherpa PUNC 模型: {}", punc_files.model.display()))?;
    writeln!(writer, "sherpa_asr=loadable")?;
    writeln!(writer, "sherpa_vad=loadable")?;
    writeln!(writer, "sherpa_punc=loadable")?;
    Ok(())
}
