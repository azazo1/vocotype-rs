use std::path::Path;
use std::time::Duration;

use anyhow::{Result, anyhow};

pub(super) fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("路径不是有效 UTF-8: {}", path.display()))
}

pub(super) fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| *sample as f32 / i16::MAX as f32)
        .collect()
}

pub(super) fn samples_to_ms(samples: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    Duration::from_secs_f64(samples as f64 / sample_rate as f64).as_millis() as u32
}
