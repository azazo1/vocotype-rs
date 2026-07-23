use std::path::Path;

use anyhow::{Result, bail};
use iflytek_core::{
    FRAME_LENGTH, FRAME_SHIFT, OriginalFeatureExtractor, OriginalVadEndpoint,
    VAD_FEATURE_SIZE, VadEndpointConfig, VadStatus,
};
use ort::session::Session;
use ort::value::Tensor;

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
}

impl EdgeEsrVad {
    pub fn load(path: &Path, config: EdgeEsrVadConfig) -> Result<Self> {
        if config.sample_rate != 16_000 {
            bail!("EdgeEsr VAD requires a 16000 Hz sample rate")
        }
        super::init_onnx_runtime_api()?;
        let session = super::load_plain_session(path)?;
        let extractor = OriginalFeatureExtractor::with_feature_size(VAD_FEATURE_SIZE)?;
        let endpoint = OriginalVadEndpoint::new(VadEndpointConfig {
            vad_threshold: config.threshold,
            ..VadEndpointConfig::default()
        })?;
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
    }

    pub fn push(&mut self, samples: &[i16]) -> Result<Vec<EdgeEsrVadSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        self.samples.extend_from_slice(samples);
        self.process_available_frames()
    }

    pub fn finish(&mut self) -> Result<Vec<EdgeEsrVadSegment>> {
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
        Ok(output)
    }

    pub fn detected(&self) -> bool {
        self.active_segment_start.is_some()
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
            let q22 = self.extractor.extract_frame_q22(frame)?;
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
        let outputs = self.session.run(ort::inputs! {
            "stacked_features" => Tensor::from_array(([1usize, VAD_INPUT_SIZE], features))?,
            "cell_in" => Tensor::from_array(([1usize, CELL_SIZE], self.neural.cell.clone()))?,
            "projection_in" => Tensor::from_array(([1usize, PROJECTION_SIZE], self.neural.projection.clone()))?,
            "context_in" => Tensor::from_array(([1usize, CONTEXT_FRAMES, PROJECTION_SIZE], self.neural.context.clone()))?,
            "count_in" => Tensor::from_array(([1usize], self.neural.count.clone()))?,
        })?;

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
