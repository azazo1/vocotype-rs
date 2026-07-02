#[derive(Clone, Debug)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub threshold: f32,
    pub pre_roll_ms: u32,
    pub tail_padding_ms: u32,
    pub end_silence_ms: u32,
    pub min_speech_ms: u32,
    pub max_segment_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            threshold: 0.5,
            pre_roll_ms: 180,
            tail_padding_ms: 180,
            end_silence_ms: 650,
            min_speech_ms: 240,
            max_segment_ms: 15_000,
        }
    }
}

impl VadConfig {
    pub(super) fn samples_for_ms(&self, ms: u32) -> usize {
        (self.sample_rate as usize * ms as usize) / 1_000
    }
}
