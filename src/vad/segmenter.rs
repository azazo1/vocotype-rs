use std::path::Path;

use anyhow::{Result, anyhow};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use super::audio::{i16_to_f32, path_string, samples_to_ms};
use super::config::VadConfig;
use super::segment::{SegmentReason, SpeechSegment, expand_bounds};

pub struct VadSegmenter {
    config: VadConfig,
    detector: VoiceActivityDetector,
    input_samples: Vec<i16>,
    input_offset: usize,
    emitted_until: usize,
}

impl VadSegmenter {
    pub fn new(config: VadConfig, model_path: &Path) -> Result<Self> {
        let vad_config = VadModelConfig {
            sample_rate: config.sample_rate as i32,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            silero_vad: SileroVadModelConfig {
                model: Some(path_string(model_path)?),
                threshold: config.threshold,
                min_silence_duration: config.end_silence_ms as f32 / 1_000.0,
                min_speech_duration: config.min_speech_ms as f32 / 1_000.0,
                window_size: 512,
                max_speech_duration: config.max_segment_ms as f32 / 1_000.0,
            },
            ..Default::default()
        };

        let buffer_seconds = (config.max_segment_ms as f32 / 1_000.0 + 5.0).max(10.0);
        let detector = VoiceActivityDetector::create(&vad_config, buffer_seconds)
            .ok_or_else(|| anyhow!("无法加载 sherpa VAD 模型: {}", model_path.display()))?;

        Ok(Self {
            config,
            detector,
            input_samples: Vec::new(),
            input_offset: 0,
            emitted_until: 0,
        })
    }

    pub fn reset(&mut self) {
        self.detector.clear();
        self.detector.reset();
        self.input_samples.clear();
        self.input_offset = 0;
        self.emitted_until = 0;
    }

    pub fn push(&mut self, frame: &[i16]) -> Vec<SpeechSegment> {
        if frame.is_empty() {
            return Vec::new();
        }

        self.input_samples.extend_from_slice(frame);
        let samples = i16_to_f32(frame);
        self.detector.accept_waveform(&samples);
        self.drain_segments(SegmentReason::EndSilence)
    }

    pub fn finish(&mut self) -> Vec<SpeechSegment> {
        self.detector.flush();
        self.drain_segments(SegmentReason::Finish)
    }

    pub fn detected(&self) -> bool {
        self.detector.detected()
    }

    fn drain_segments(&mut self, default_reason: SegmentReason) -> Vec<SpeechSegment> {
        let mut output = Vec::new();
        while let Some(front) = self.detector.front() {
            let start = front.start().max(0) as usize;
            let speech_len = front.n().max(0) as usize;
            let reason = if matches!(default_reason, SegmentReason::EndSilence)
                && speech_len >= self.config.samples_for_ms(self.config.max_segment_ms)
            {
                SegmentReason::MaxDuration
            } else {
                default_reason.clone()
            };
            drop(front);
            self.detector.pop();

            let Some(segment) = self.segment_from_bounds(start, speech_len, reason) else {
                continue;
            };
            output.push(segment);
        }
        output
    }

    fn segment_from_bounds(
        &mut self,
        speech_start: usize,
        speech_len: usize,
        reason: SegmentReason,
    ) -> Option<SpeechSegment> {
        if speech_len == 0 {
            return None;
        }

        let speech_end = speech_start.saturating_add(speech_len);
        let available_end = self.input_offset + self.input_samples.len();
        let (expanded_start, expanded_end) = expand_bounds(
            speech_start,
            speech_end,
            self.config.samples_for_ms(self.config.pre_roll_ms),
            self.config.samples_for_ms(self.config.tail_padding_ms),
            available_end,
        );
        let start = expanded_start.max(self.emitted_until);
        let end = expanded_end.max(start).min(available_end);
        if end <= start || start < self.input_offset {
            return None;
        }

        let local_start = start - self.input_offset;
        let local_end = end - self.input_offset;
        let samples = self.input_samples[local_start..local_end].to_vec();
        self.emitted_until = end;
        self.prune_buffer();

        Some(SpeechSegment {
            samples,
            reason,
            speech_ms: samples_to_ms(speech_len, self.config.sample_rate),
        })
    }

    fn prune_buffer(&mut self) {
        let keep_from = self
            .emitted_until
            .saturating_sub(self.config.samples_for_ms(self.config.pre_roll_ms));
        if keep_from <= self.input_offset {
            return;
        }
        let drain_len = (keep_from - self.input_offset).min(self.input_samples.len());
        self.input_samples.drain(..drain_len);
        self.input_offset += drain_len;
    }
}
