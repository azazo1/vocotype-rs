use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Serialize;
use uuid::Uuid;

use crate::asr::TranscriptionResult;

#[derive(Clone, Debug)]
pub struct DatasetRecorder {
    root: PathBuf,
}

#[derive(Debug, Serialize)]
struct DatasetRecord {
    id: String,
    audio: String,
    text: String,
    raw_text: String,
    duration: f32,
    sample_rate: u32,
    inference_latency: f32,
    confidence: f32,
    timestamp: String,
}

impl DatasetRecorder {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(root.join("audio"))?;
        Ok(Self { root })
    }

    pub fn record(&self, result: &TranscriptionResult, sample_rate: u32, samples: &[i16]) -> Result<()> {
        if !result.success {
            return Ok(());
        }

        let id = format!(
            "{}-{}",
            Utc::now().format("%Y%m%d-%H%M%S-%3f"),
            Uuid::new_v4().simple()
        );
        let audio_name = format!("{}.wav", id);
        let audio_rel = Path::new("audio").join(&audio_name);
        let audio_path = self.root.join(&audio_rel);
        crate::wav::write_wav_mono_i16(&audio_path, sample_rate, samples)?;

        let record = DatasetRecord {
            id,
            audio: audio_rel.to_string_lossy().replace('\\', "/"),
            text: result.text.clone(),
            raw_text: result.raw_text.clone(),
            duration: result.duration,
            sample_rate,
            inference_latency: result.inference_latency,
            confidence: result.confidence,
            timestamp: Utc::now().to_rfc3339(),
        };

        let line = serde_json::to_string(&record)?;
        let jsonl = self.root.join("dataset.jsonl");
        crate::app::ensure_parent(&jsonl)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl)
            .with_context(|| format!("无法打开数据集文件: {}", jsonl.display()))?;
        use std::io::Write;
        writeln!(file, "{}", line)?;
        Ok(())
    }
}
