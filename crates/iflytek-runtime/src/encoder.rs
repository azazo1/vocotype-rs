use std::collections::BTreeMap;

use anyhow::{Result, bail};
use half::f16;
use iflytek_core::{FEATURE_SIZE, FRAME_LENGTH, FRAME_SHIFT, OriginalFeatureExtractor};
use ort::session::Session;
use ort::value::Tensor;

use crate::tensor::{TensorData, output_f16, output_f32};

const HIDDEN_SIZE: usize = 512;
const VGG_CHUNK_FRAMES: usize = 68;
const VGG_STRIDE_FRAMES: usize = 64;
const VGG_OUTPUT_FRAMES: usize = 32;
const VGG_STARTUP_BLANKS: usize = 2;
const CONFORMER_WINDOW_FRAMES: usize = 64;
const CONFORMER_STRIDE_FRAMES: usize = 32;
const CONFORMER_OUTPUT_CHUNKS: usize = 2;
pub(crate) const ENCODER_CHUNK_FRAMES: usize = 16;
const CONFORMER_STARTUP_BLANKS: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct EncoderChunk {
    pub(crate) conformer_output: Vec<f32>,
    pub(crate) conformer_mask: Vec<f32>,
    pub(crate) at_h: Vec<f32>,
    pub(crate) chunk_at_h: Vec<f32>,
    pub(crate) enc_mem_h: Vec<f32>,
}

pub(crate) struct EdgeEsrEncoder<'a> {
    vgg: &'a mut Session,
    conformer: &'a mut Session,
    extractor: OriginalFeatureExtractor,
    vgg_state: BTreeMap<String, TensorData>,
    conformer_state: BTreeMap<String, TensorData>,
    pcm_buffer: Vec<i16>,
    pcm_cursor: usize,
    feature_buffer: Vec<f32>,
    mask_buffer: Vec<f32>,
    vgg_output_buffer: Vec<f16>,
    vgg_mask_buffer: Vec<f16>,
    pending_conformer_chunks: Vec<EncoderChunk>,
    vgg_calls: usize,
    conformer_calls: usize,
    finalized: bool,
}

impl<'a> EdgeEsrEncoder<'a> {
    pub(crate) fn new(vgg: &'a mut Session, conformer: &'a mut Session) -> Result<Self> {
        validate_names(vgg, &["vgg_input", "vgg_mask"], "VGG")?;
        validate_names(conformer, &["conformer_input", "conformer_mask"], "Conformer")?;
        let mut vgg_state = BTreeMap::new();
        for index in 0..4 {
            let name = format!("vgg_his_in_{}", index);
            vgg_state.insert(name.clone(), TensorData::zeros_for_input(vgg, &name)?);
        }
        let mut conformer_state = BTreeMap::new();
        for family in ["conv", "k", "v"] {
            for index in 0..16 {
                let name = format!("conformer_{}_his_in_{}", family, index);
                conformer_state.insert(
                    name.clone(),
                    TensorData::zeros_for_input(conformer, &name)?,
                );
            }
        }
        Ok(Self {
            vgg,
            conformer,
            extractor: OriginalFeatureExtractor::default(),
            vgg_state,
            conformer_state,
            pcm_buffer: Vec::new(),
            pcm_cursor: 0,
            feature_buffer: Vec::new(),
            mask_buffer: Vec::new(),
            vgg_output_buffer: Vec::new(),
            vgg_mask_buffer: Vec::new(),
            pending_conformer_chunks: Vec::new(),
            vgg_calls: 0,
            conformer_calls: 0,
            finalized: false,
        })
    }

    pub(crate) fn prime_startup(&mut self) -> Result<Vec<EncoderChunk>> {
        if self.vgg_calls != 0 || self.conformer_calls != 0 || !self.feature_buffer.is_empty() {
            bail!("EdgeEsr encoder startup can only be primed after reset")
        }
        let feature_frames = VGG_CHUNK_FRAMES + VGG_STRIDE_FRAMES;
        let sample_count = FRAME_LENGTH + (feature_frames - 1) * FRAME_SHIFT;
        self.accept_pcm(&vec![0; sample_count], false)
    }

    pub(crate) fn accept_pcm(&mut self, samples: &[i16], final_input: bool) -> Result<Vec<EncoderChunk>> {
        if self.finalized {
            bail!("EdgeEsr encoder is already finalized")
        }
        self.pcm_buffer.extend_from_slice(samples);
        let mut features = Vec::new();
        while self.pcm_buffer.len().saturating_sub(self.pcm_cursor) >= FRAME_LENGTH {
            let frame = &self.pcm_buffer[self.pcm_cursor..self.pcm_cursor + FRAME_LENGTH];
            features.extend(self.extractor.extract_frame(frame)?);
            self.pcm_cursor += FRAME_SHIFT;
        }
        if self.pcm_cursor >= FRAME_LENGTH {
            self.pcm_buffer.drain(..self.pcm_cursor);
            self.pcm_cursor = 0;
        }
        if final_input {
            self.pcm_buffer.clear();
            self.pcm_cursor = 0;
        }
        self.accept_features(&features, final_input)
    }

    fn accept_features(&mut self, features: &[f32], final_input: bool) -> Result<Vec<EncoderChunk>> {
        if !features.len().is_multiple_of(FEATURE_SIZE) {
            bail!("EdgeEsr feature buffer has an invalid shape")
        }
        let frames = features.len() / FEATURE_SIZE;
        self.feature_buffer.extend_from_slice(features);
        self.mask_buffer.extend(std::iter::repeat_n(1.0, frames));
        let mut output = Vec::new();
        while self.feature_buffer.len() / FEATURE_SIZE >= VGG_CHUNK_FRAMES {
            let feature_count = VGG_CHUNK_FRAMES * FEATURE_SIZE;
            let chunk_features = self.feature_buffer[..feature_count].to_vec();
            let chunk_mask = self.mask_buffer[..VGG_CHUNK_FRAMES].to_vec();
            output.extend(self.run_vgg_chunk(&chunk_features, &chunk_mask, true)?);
            self.feature_buffer.drain(..VGG_STRIDE_FRAMES * FEATURE_SIZE);
            self.mask_buffer.drain(..VGG_STRIDE_FRAMES);
        }

        if final_input {
            let remaining = self.feature_buffer.len() / FEATURE_SIZE;
            let overlap = VGG_CHUNK_FRAMES - VGG_STRIDE_FRAMES;
            let has_unseen = remaining > if self.vgg_calls == 0 { 0 } else { overlap };
            if has_unseen {
                let mut padded_features = vec![0.0; VGG_CHUNK_FRAMES * FEATURE_SIZE];
                let mut padded_mask = vec![0.0; VGG_CHUNK_FRAMES];
                padded_features[..self.feature_buffer.len()]
                    .copy_from_slice(&self.feature_buffer);
                padded_mask[..remaining].copy_from_slice(&self.mask_buffer);
                let _ = self.run_vgg_chunk(&padded_features, &padded_mask, false)?;
            }
            self.feature_buffer.clear();
            self.mask_buffer.clear();
            output.append(&mut self.pending_conformer_chunks);
            self.finalized = true;
        }
        Ok(output)
    }

    fn run_vgg_chunk(
        &mut self,
        features: &[f32],
        mask: &[f32],
        publish: bool,
    ) -> Result<Vec<EncoderChunk>> {
        let mut inputs = Vec::with_capacity(2 + self.vgg_state.len());
        inputs.push((
            "vgg_input".to_string(),
            Tensor::from_array((
                vec![1, VGG_CHUNK_FRAMES, FEATURE_SIZE, 1],
                features.to_vec(),
            ))?
            .into_dyn(),
        ));
        inputs.push((
            "vgg_mask".to_string(),
            Tensor::from_array((vec![1, VGG_CHUNK_FRAMES, 1, 1], mask.to_vec()))?
                .into_dyn(),
        ));
        for (name, state) in self.vgg_state.clone() {
            inputs.push((name, state.into_value()?));
        }
        let outputs = self.vgg.run(inputs)?;
        let (_, mut vgg_output) = output_f16(&outputs["vgg_output"], "vgg_output")?;
        let (_, mut vgg_mask) = output_f16(&outputs["vgg_mask_output"], "vgg_mask_output")?;
        for index in 0..4 {
            let input_name = format!("vgg_his_in_{}", index);
            let output_name = format!("vgg_his_out_{}", index);
            self.vgg_state.insert(
                input_name,
                TensorData::from_value(&outputs[output_name.as_str()], &output_name)?,
            );
        }
        drop(outputs);
        self.vgg_calls += 1;
        if !publish {
            return Ok(Vec::new());
        }
        if self.vgg_calls <= VGG_STARTUP_BLANKS {
            vgg_output.fill(f16::ZERO);
            vgg_mask.fill(f16::ZERO);
        }
        if vgg_output.len() != VGG_OUTPUT_FRAMES * HIDDEN_SIZE
            || vgg_mask.len() != VGG_OUTPUT_FRAMES
        {
            bail!("VGG encoder output shape does not match EdgeEsr contract")
        }
        self.vgg_output_buffer.extend(vgg_output);
        self.vgg_mask_buffer.extend(vgg_mask);
        self.run_conformer_windows()
    }

    fn run_conformer_windows(&mut self) -> Result<Vec<EncoderChunk>> {
        let mut results = Vec::new();
        while self.vgg_output_buffer.len() / HIDDEN_SIZE >= CONFORMER_WINDOW_FRAMES {
            let input_size = CONFORMER_WINDOW_FRAMES * HIDDEN_SIZE;
            let input = self.vgg_output_buffer[..input_size].to_vec();
            let mask = self.vgg_mask_buffer[..CONFORMER_WINDOW_FRAMES].to_vec();
            let mut inputs = Vec::with_capacity(2 + self.conformer_state.len());
            inputs.push((
                "conformer_input".to_string(),
                Tensor::from_array((
                    vec![1, 1, CONFORMER_WINDOW_FRAMES, HIDDEN_SIZE],
                    input,
                ))?
                .into_dyn(),
            ));
            inputs.push((
                "conformer_mask".to_string(),
                Tensor::from_array((vec![1, 1, CONFORMER_WINDOW_FRAMES, 1], mask))?
                    .into_dyn(),
            ));
            for (name, state) in self.conformer_state.clone() {
                inputs.push((name, state.into_value()?));
            }
            let outputs = self.conformer.run(inputs)?;
            let (_, conformer_output) = output_f32(&outputs["conformer_output"], "conformer_output")?;
            let (_, conformer_mask) = output_f32(&outputs["conformer_mask_out"], "conformer_mask_out")?;
            let (_, at_h) = output_f32(&outputs["at_h"], "at_h")?;
            let (_, chunk_at_h) = output_f32(&outputs["chunk_at_h"], "chunk_at_h")?;
            let (_, enc_mem_h) = output_f32(&outputs["enc_mem_h"], "enc_mem_h")?;
            for family in ["conv", "k", "v"] {
                for index in 0..16 {
                    let input_name = format!("conformer_{}_his_in_{}", family, index);
                    let output_name = format!("conformer_{}_his_out_{}", family, index);
                    self.conformer_state.insert(
                        input_name,
                        TensorData::from_value(
                            &outputs[output_name.as_str()],
                            &output_name,
                        )?,
                    );
                }
            }
            drop(outputs);
            self.conformer_calls += 1;
            let blank = self.conformer_calls <= CONFORMER_STARTUP_BLANKS;
            let chunks = split_conformer_output(
                &conformer_output,
                &conformer_mask,
                &at_h,
                &chunk_at_h,
                &enc_mem_h,
                blank,
            )?;
            results.push(chunks[0].clone());
            self.pending_conformer_chunks = vec![chunks[1].clone()];
            self.vgg_output_buffer.drain(..CONFORMER_STRIDE_FRAMES * HIDDEN_SIZE);
            self.vgg_mask_buffer.drain(..CONFORMER_STRIDE_FRAMES);
        }
        Ok(results)
    }
}

fn split_conformer_output(
    output: &[f32],
    mask: &[f32],
    at_h: &[f32],
    chunk_at_h: &[f32],
    enc_mem_h: &[f32],
    blank: bool,
) -> Result<[EncoderChunk; 2]> {
    let tensor_size = CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
    if [output.len(), at_h.len(), chunk_at_h.len(), enc_mem_h.len()]
        .iter()
        .any(|length| *length != tensor_size)
        || mask.len() != CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES
    {
        bail!("Conformer output shape does not match EdgeEsr contract")
    }
    Ok(std::array::from_fn(|chunk| {
        let hidden_start = chunk * ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
        let hidden_end = hidden_start + ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
        let mask_start = chunk * ENCODER_CHUNK_FRAMES;
        let mask_end = mask_start + ENCODER_CHUNK_FRAMES;
        let select = |values: &[f32]| {
            if blank {
                vec![0.0; hidden_end - hidden_start]
            } else {
                values[hidden_start..hidden_end].to_vec()
            }
        };
        EncoderChunk {
            conformer_output: select(output),
            conformer_mask: if blank {
                vec![0.0; ENCODER_CHUNK_FRAMES]
            } else {
                mask[mask_start..mask_end].to_vec()
            },
            at_h: select(at_h),
            chunk_at_h: select(chunk_at_h),
            enc_mem_h: select(enc_mem_h),
        }
    }))
}

fn validate_names(session: &Session, required: &[&str], label: &str) -> Result<()> {
    for name in required {
        if !session.inputs().iter().any(|input| input.name() == *name) {
            bail!("{} encoder is missing input {}", label, name)
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformer_output_splits_into_consecutive_chunks() {
        let hidden_length = CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
        let hidden = (0..hidden_length).map(|value| value as f32).collect::<Vec<_>>();
        let mask = (0..CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let chunks = split_conformer_output(
            &hidden,
            &mask,
            &hidden,
            &hidden,
            &hidden,
            false,
        )
        .unwrap();
        assert_eq!(chunks[0].conformer_output[0], 0.0);
        assert_eq!(
            chunks[1].conformer_output[0],
            (ENCODER_CHUNK_FRAMES * HIDDEN_SIZE) as f32
        );
        assert_eq!(chunks[0].conformer_mask[0], 0.0);
        assert_eq!(chunks[1].conformer_mask[0], ENCODER_CHUNK_FRAMES as f32);
    }

    #[test]
    fn startup_chunk_blanks_every_attention_tensor() {
        let hidden_length = CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
        let hidden = vec![1.0; hidden_length];
        let mask = vec![1.0; CONFORMER_OUTPUT_CHUNKS * ENCODER_CHUNK_FRAMES];
        let chunks = split_conformer_output(
            &hidden,
            &mask,
            &hidden,
            &hidden,
            &hidden,
            true,
        )
        .unwrap();
        for chunk in chunks {
            assert!(chunk.conformer_output.iter().all(|value| *value == 0.0));
            assert!(chunk.conformer_mask.iter().all(|value| *value == 0.0));
            assert!(chunk.at_h.iter().all(|value| *value == 0.0));
            assert!(chunk.chunk_at_h.iter().all(|value| *value == 0.0));
            assert!(chunk.enc_mem_h.iter().all(|value| *value == 0.0));
        }
    }
}
