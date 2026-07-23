use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use iflytek_core::{
    FRAME_LENGTH, FRAME_SHIFT, OriginalFeatureExtractor, OriginalVadEndpoint,
    VAD_FEATURE_SIZE, VadEndpointConfig, VadStatus,
};
use ort::session::Session;
use ort::value::Tensor;
use tracing::{debug, info, trace};

const STACKED_FRAMES: usize = 4;
const VAD_INPUT_SIZE: usize = VAD_FEATURE_SIZE * STACKED_FRAMES;
const CELL_SIZE: usize = 288;
const PROJECTION_SIZE: usize = 96;
const CONTEXT_FRAMES: usize = 3;
const SAMPLES_PER_FRAME: usize = 160;

#[derive(Clone, Debug)]
pub struct EdgeEsrVadConfig {
    pub sample_rate: u32,
    pub threshold: f32,
    pub pre_roll_ms: u32,
    pub tail_padding_ms: u32,
    pub end_silence_ms: u32,
    pub min_speech_ms: u32,
    pub max_segment_ms: u32,
}

impl Default for EdgeEsrVadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            threshold: 0.0,
            pre_roll_ms: 300,
            tail_padding_ms: 300,
            end_silence_ms: 3_000,
            min_speech_ms: 0,
            max_segment_ms: 60_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EdgeEsrVadSegmentReason {
    EndSilence,
    MaxDuration,
    Finish,
}

#[derive(Clone, Debug)]
pub struct EdgeEsrVadSegment {
    pub samples: Vec<i16>,
    pub reason: EdgeEsrVadSegmentReason,
    pub speech_ms: u32,
    pub start_sample: usize,
    pub end_sample: usize,
    pub audio_start_sample: usize,
    pub audio_end_sample: usize,
}

struct NeuralState {
    cell: Vec<f32>,
    projection: Vec<f32>,
    context: Vec<f32>,
    count: Vec<i64>,
}

impl Default for NeuralState {
    fn default() -> Self {
        Self {
            cell: vec![0.0; CELL_SIZE],
            projection: vec![0.0; PROJECTION_SIZE],
            context: vec![0.0; CONTEXT_FRAMES * PROJECTION_SIZE],
            count: vec![0],
        }
    }
}

pub struct EdgeEsrVad {
    config: EdgeEsrVadConfig,
    session: Session,
    extractor: OriginalFeatureExtractor,
    neural: NeuralState,
    samples: Vec<i16>,
    feature_cursor: usize,
    feature_group: Vec<i16>,
    pending_frames: Option<Vec<usize>>,
    feature_frames: usize,
    endpoint: OriginalVadEndpoint,
    active_segment_start: Option<i32>,
    active_speech_start: Option<i32>,
    stream_started: Option<Instant>,
    frontend_elapsed: Duration,
    onnx_elapsed: Duration,
    inference_groups: usize,
}

impl EdgeEsrVad {
    pub fn load(path: &Path, config: EdgeEsrVadConfig) -> Result<Self> {
        if config.sample_rate != 16_000 {
            bail!("EdgeEsr VAD requires a 16000 Hz sample rate")
        }
        super::init_onnx_runtime_api()?;
        let session = super::load_plain_session(path, super::default_intra_threads())?;
        let extractor = OriginalFeatureExtractor::with_feature_size(VAD_FEATURE_SIZE)?;
        let endpoint = OriginalVadEndpoint::new(endpoint_config(&config))?;
        let runtime = Self {
            config,
            session,
            extractor,
            neural: NeuralState::default(),
            samples: Vec::new(),
            feature_cursor: 0,
            feature_group: Vec::with_capacity(VAD_INPUT_SIZE),
            pending_frames: None,
            feature_frames: 0,
            endpoint,
            active_segment_start: None,
            active_speech_start: None,
            stream_started: None,
            frontend_elapsed: Duration::ZERO,
            onnx_elapsed: Duration::ZERO,
            inference_groups: 0,
        };
        runtime.validate_model()?;
        Ok(runtime)
    }

    pub fn reset(&mut self) {
        self.neural = NeuralState::default();
        self.samples.clear();
        self.feature_cursor = 0;
        self.feature_group.clear();
        self.pending_frames = None;
        self.feature_frames = 0;
        self.endpoint.reset();
        self.active_segment_start = None;
        self.active_speech_start = None;
        self.stream_started = None;
        self.frontend_elapsed = Duration::ZERO;
        self.onnx_elapsed = Duration::ZERO;
        self.inference_groups = 0;
    }

    pub fn push(&mut self, samples: &[i16]) -> Result<Vec<EdgeEsrVadSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        self.stream_started.get_or_insert_with(Instant::now);
        self.samples.extend_from_slice(samples);
        self.process_available_frames()
    }

    pub fn finish(&mut self) -> Result<Vec<EdgeEsrVadSegment>> {
        let started = *self.stream_started.get_or_insert_with(Instant::now);
        let mut output = self.process_available_frames()?;
        let rounded_frames = self.feature_frames.div_ceil(STACKED_FRAMES) * STACKED_FRAMES;
        if !self.feature_group.is_empty() {
            let real_frames = self.feature_group.len() / VAD_FEATURE_SIZE;
            self.feature_group.resize(VAD_INPUT_SIZE, 0);
            let frames = (0..real_frames.saturating_sub(1))
                .map(|index| self.feature_cursor / FRAME_SHIFT - real_frames + index)
                .collect::<Vec<_>>();
            output.extend(self.run_group(frames, rounded_frames as i32)?);
        } else if let Some(pending) = &mut self.pending_frames {
            let _ = pending.pop();
        }
        if self.feature_frames > 0 {
            output.extend(self.run_group(Vec::new(), rounded_frames as i32)?);
            let missing = rounded_frames
                .saturating_sub(self.endpoint.state().checked_frame_count as usize);
            for _ in 0..missing {
                let evidence = self.endpoint.pad_silence();
                let status = self.endpoint.step(evidence.frame, rounded_frames as i32)?;
                if let Some(segment) = self.record_status(status, true) {
                    output.push(segment);
                }
                self.endpoint.state_mut().current_frame = evidence.frame + 1;
                self.endpoint.state_mut().read_delay = 0;
            }
        }
        let status = self.endpoint.finalize(self.endpoint.state().current_frame);
        if status == VadStatus::SpeechEnd
            && let Some(segment) = self.record_status(status, true)
        {
            output.push(segment);
        }
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            frontend_ms = self.frontend_elapsed.as_millis(),
            onnx_ms = self.onnx_elapsed.as_millis(),
            feature_frames = self.feature_frames,
            inference_groups = self.inference_groups,
            segment_count = output.len(),
            "EdgeEsr VAD completed"
        );
        Ok(output)
    }

    pub fn detected(&self) -> bool {
        self.active_segment_start.is_some()
    }

    pub fn active_audio_start_sample(&self) -> Option<usize> {
        self.active_segment_start
            .map(|frame| frame.max(0) as usize * SAMPLES_PER_FRAME)
    }

    fn validate_model(&self) -> Result<()> {
        let input_names = self
            .session
            .inputs()
            .iter()
            .map(|input| input.name())
            .collect::<Vec<_>>();
        for required in [
            "stacked_features",
            "cell_in",
            "projection_in",
            "context_in",
            "count_in",
        ] {
            if !input_names.contains(&required) {
                bail!("EdgeEsr VAD model is missing input {}", required)
            }
        }
        let output_names = self
            .session
            .outputs()
            .iter()
            .map(|output| output.name())
            .collect::<Vec<_>>();
        for required in [
            "logits",
            "ready",
            "cell_out",
            "projection_out",
            "context_out",
            "count_out",
        ] {
            if !output_names.contains(&required) {
                bail!("EdgeEsr VAD model is missing output {}", required)
            }
        }
        Ok(())
    }

    fn process_available_frames(&mut self) -> Result<Vec<EdgeEsrVadSegment>> {
        let mut output = Vec::new();
        while self.samples.len().saturating_sub(self.feature_cursor) >= FRAME_LENGTH {
            let frame = &self.samples[self.feature_cursor..self.feature_cursor + FRAME_LENGTH];
            let frontend_started = Instant::now();
            let q22 = self.extractor.extract_frame_q22(frame)?;
            self.frontend_elapsed += frontend_started.elapsed();
            self.feature_group
                .extend(q22.into_iter().map(|value| (value >> 11) as i16));
            let frame_index = self.feature_cursor / FRAME_SHIFT;
            self.feature_cursor += FRAME_SHIFT;
            self.feature_frames += 1;
            if self.feature_group.len() == VAD_INPUT_SIZE {
                let start = frame_index + 1 - STACKED_FRAMES;
                output.extend(self.run_group((start..=frame_index).collect(), -1)?);
            }
        }
        Ok(output)
    }

    fn run_group(
        &mut self,
        frames: Vec<usize>,
        flush_frame: i32,
    ) -> Result<Vec<EdgeEsrVadSegment>> {
        let features = if self.feature_group.is_empty() {
            vec![0_i16; VAD_INPUT_SIZE]
        } else {
            std::mem::take(&mut self.feature_group)
        };
        let onnx_started = Instant::now();
        let outputs = self.session.run(ort::inputs! {
            "stacked_features" => Tensor::from_array(([1usize, VAD_INPUT_SIZE], features))?,
            "cell_in" => Tensor::from_array(([1usize, CELL_SIZE], self.neural.cell.clone()))?,
            "projection_in" => Tensor::from_array(([1usize, PROJECTION_SIZE], self.neural.projection.clone()))?,
            "context_in" => Tensor::from_array(([1usize, CONTEXT_FRAMES, PROJECTION_SIZE], self.neural.context.clone()))?,
            "count_in" => Tensor::from_array(([1usize], self.neural.count.clone()))?,
        })?;
        let onnx_elapsed = onnx_started.elapsed();
        self.onnx_elapsed += onnx_elapsed;
        self.inference_groups += 1;
        trace!(
            group = self.inference_groups,
            elapsed_us = onnx_elapsed.as_micros(),
            "EdgeEsr VAD inference group completed"
        );

        let (_, cell) = outputs["cell_out"].try_extract_tensor::<f32>()?;
        let (_, projection) = outputs["projection_out"].try_extract_tensor::<f32>()?;
        let (_, context) = outputs["context_out"].try_extract_tensor::<f32>()?;
        let (_, count) = outputs["count_out"].try_extract_tensor::<i64>()?;
        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (_, ready) = outputs["ready"].try_extract_tensor::<bool>()?;
        self.neural = NeuralState {
            cell: cell.to_vec(),
            projection: projection.to_vec(),
            context: context.to_vec(),
            count: count.to_vec(),
        };
        let logits = logits.to_vec();
        let ready = ready.first().copied().unwrap_or(false);
        drop(outputs);

        let previous_frames = self.pending_frames.replace(frames);
        if !ready {
            return Ok(Vec::new());
        }
        let Some(previous_frames) = previous_frames else {
            return Ok(Vec::new());
        };
        let frame_count = previous_frames.len().min(logits.len() / 2);
        let mut output = Vec::new();
        for (index, frame) in previous_frames.into_iter().take(frame_count).enumerate() {
            let energy = frame_energy_active(&self.samples, frame);
            let evidence = self.endpoint.push_logits(
                logits[index * 2],
                logits[index * 2 + 1],
                energy,
            );
            let status = self.endpoint.step(evidence.frame, flush_frame)?;
            if let Some(segment) = self.record_status(status, flush_frame >= 0) {
                output.push(segment);
            }
            self.endpoint.state_mut().current_frame = evidence.frame + 1;
            self.endpoint.state_mut().read_delay = 0;
        }
        Ok(output)
    }

    fn record_status(
        &mut self,
        status: VadStatus,
        final_flush: bool,
    ) -> Option<EdgeEsrVadSegment> {
        if status == VadStatus::SpeechStart {
            self.active_segment_start = Some(self.endpoint.state().start_pause_frame);
            self.active_speech_start = Some(self.endpoint.state().real_pause_frame);
            return None;
        }
        if status != VadStatus::SpeechEnd {
            return None;
        }
        let segment_start = self.active_segment_start.take()?;
        let speech_start = self.active_speech_start.take()?;
        let end_frame = self.endpoint.state().end_pause_frame + 1;
        let speech_end_frame = self.endpoint.state().real_pause_frame.max(speech_start);
        let audio_start = segment_start.max(0) as usize * SAMPLES_PER_FRAME;
        let audio_end = end_frame.max(segment_start) as usize * SAMPLES_PER_FRAME;
        let speech_start_sample = speech_start.max(0) as usize * SAMPLES_PER_FRAME;
        let speech_end_sample = speech_end_frame as usize * SAMPLES_PER_FRAME;
        let audio_start = audio_start.min(self.samples.len());
        let audio_end = audio_end.max(audio_start).min(self.samples.len());
        let speech_start_sample = speech_start_sample.min(self.samples.len());
        let speech_end_sample = speech_end_sample.max(speech_start_sample).min(self.samples.len());
        let speech_len = speech_end_sample.saturating_sub(speech_start_sample);
        if audio_end <= audio_start
            || speech_len == 0
            || speech_len < self.samples_for_ms(self.config.min_speech_ms)
        {
            debug!(
                audio_start,
                audio_end,
                speech_start_sample,
                speech_end_sample,
                "EdgeEsr VAD 丢弃空片段或过短片段"
            );
            return None;
        }
        let reason = if final_flush {
            EdgeEsrVadSegmentReason::Finish
        } else if speech_len >= self.samples_for_ms(self.config.max_segment_ms) {
            EdgeEsrVadSegmentReason::MaxDuration
        } else {
            EdgeEsrVadSegmentReason::EndSilence
        };
        Some(EdgeEsrVadSegment {
            samples: self.samples[audio_start..audio_end].to_vec(),
            reason,
            speech_ms: ((speech_len as u64 * 1_000) / self.config.sample_rate as u64) as u32,
            start_sample: speech_start_sample,
            end_sample: speech_end_sample,
            audio_start_sample: audio_start,
            audio_end_sample: audio_end,
        })
    }

    fn samples_for_ms(&self, milliseconds: u32) -> usize {
        self.config.sample_rate as usize * milliseconds as usize / 1_000
    }
}

fn endpoint_config(config: &EdgeEsrVadConfig) -> VadEndpointConfig {
    VadEndpointConfig {
        frame_start_margin: milliseconds_to_frames(config.pre_roll_ms, config.sample_rate),
        frame_end_margin: milliseconds_to_frames(config.tail_padding_ms, config.sample_rate),
        end_gap: milliseconds_to_frames(config.end_silence_ms, config.sample_rate),
        vad_threshold: config.threshold,
        force_segment: milliseconds_to_frames(config.max_segment_ms, config.sample_rate),
        ..VadEndpointConfig::default()
    }
}

fn milliseconds_to_frames(milliseconds: u32, sample_rate: u32) -> i32 {
    let samples = u64::from(milliseconds).saturating_mul(u64::from(sample_rate)) / 1_000;
    let frames = samples / SAMPLES_PER_FRAME as u64;
    frames.min(i32::MAX as u64) as i32
}

fn frame_energy_active(samples: &[i16], frame: usize) -> bool {
    let start = frame * SAMPLES_PER_FRAME;
    let end = start.saturating_add(SAMPLES_PER_FRAME);
    let Some(values) = samples.get(start..end) else {
        return false;
    };
    let square_sum = values
        .iter()
        .map(|sample| i64::from(*sample) * i64::from(*sample))
        .sum::<i64>();
    let mean_square = square_sum as f32 / SAMPLES_PER_FRAME as f32;
    (mean_square + 1.0).ln() > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_config_uses_application_vad_durations() {
        let config = EdgeEsrVadConfig {
            sample_rate: 16_000,
            threshold: 0.5,
            pre_roll_ms: 180,
            tail_padding_ms: 180,
            end_silence_ms: 650,
            min_speech_ms: 240,
            max_segment_ms: 15_000,
        };

        let endpoint = endpoint_config(&config);

        assert_eq!(endpoint.frame_start_margin, 18);
        assert_eq!(endpoint.frame_end_margin, 18);
        assert_eq!(endpoint.end_gap, 65);
        assert_eq!(endpoint.force_segment, 1_500);
        assert_eq!(endpoint.vad_threshold, 0.5);
    }

    #[test]
    fn milliseconds_convert_to_ten_millisecond_frames() {
        assert_eq!(milliseconds_to_frames(0, 16_000), 0);
        assert_eq!(milliseconds_to_frames(9, 16_000), 0);
        assert_eq!(milliseconds_to_frames(10, 16_000), 1);
        assert_eq!(milliseconds_to_frames(655, 16_000), 65);
    }
}
