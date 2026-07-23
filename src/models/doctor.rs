use std::io::Write;

use anyhow::Result;
use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};

use crate::asr_backend::AsrBackend;

use super::{ModelKind, ModelStore};

pub fn write_doctor_report_for(
    store: &ModelStore,
    backend: AsrBackend,
    mut writer: impl Write,
) -> Result<()> {
    writeln!(writer, "backend={}", backend)?;
    writeln!(writer, "model_dir={}", store.paths.model_root.display())?;
    writeln!(
        writer,
        "model_cache_dir={}",
        store.paths.model_cache_root.display()
    )?;
    let revision = match backend {
        AsrBackend::Sherpa => store.revision.as_str(),
        AsrBackend::Iflytek => iflytek_runtime::MODEL_RELEASE_TAG,
    };
    writeln!(writer, "revision={}", revision)?;
    for kind in ModelKind::for_backend(backend).iter().copied() {
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
    if backend == AsrBackend::Iflytek {
        for path in store.missing_iflytek_files() {
            writeln!(writer, "missing_file={}", path.display())?;
        }
    }
    if !store.missing_models_for(backend).is_empty() {
        writeln!(writer, "hint={}", store.download_hint_for(backend))?;
    }
    Ok(())
}

pub fn loadability_report_for(
    store: &ModelStore,
    backend: AsrBackend,
    mut writer: impl Write,
) -> Result<()> {
    store.verify_required_for(backend)?;
    let _engine = crate::asr::AsrEngine::load_with_options(
        store.clone(),
        crate::asr::AsrOptions {
            backend,
            ..crate::asr::AsrOptions::default()
        },
    )?;
    if backend == AsrBackend::Iflytek {
        let vad_model = store.vad_model_path_for(backend)?;
        let _vad = iflytek_runtime::EdgeEsrVad::load(
            &vad_model,
            iflytek_runtime::EdgeEsrVadConfig::default(),
        )?;
        writeln!(writer, "iflytek_asr=loadable")?;
        writeln!(writer, "iflytek_vad=loadable")?;
        writeln!(writer, "iflytek_custom_op_domain={}", iflytek_core::CUSTOM_OP_DOMAIN)?;
        return Ok(());
    }
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
