use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use iflytek_core::{
    BeamSearchConfig, MemoryAttentionInput, MemoryAttentionResult, MemoryTryAttention,
    OriginalBeamSearch,
};
use ort::session::Session;
use ort::value::Tensor;

use crate::encoder::ENCODER_CHUNK_FRAMES;
use crate::recognizer::EncoderBank;
use crate::tensor::{TensorData, output_f32, output_i32_bits};
use crate::DecoderActiveRows;

const MAX_BEAMS: usize = 8;
const HIDDEN_SIZE: usize = 512;
const START_TOKEN_ID: i32 = 14_829;
const FRAME_BUDGET: usize = 768;
const STATE_NAMES: [&str; 3] = ["k_his_in", "v_his_in", "sum_his_in"];
const STATE_OUTPUTS: [&str; 3] = ["k_his_out", "v_his_out", "sum_his_out"];

pub(crate) struct DecoderAdvance {
    pub(crate) attention: MemoryAttentionResult,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DecoderTimings {
    pub(crate) attention: Duration,
    pub(crate) onnx: Duration,
    pub(crate) beam: Duration,
}

pub(crate) struct EdgeEsrDecoder<'a> {
    decoder1: &'a mut Session,
    decoder2: &'a mut Session,
    active_rows: DecoderActiveRows,
    attention: MemoryTryAttention,
    search: OriginalBeamSearch,
    active_count: usize,
    labels: Vec<i32>,
    lengths: Vec<i32>,
    decoder1_state: BTreeMap<String, TensorData>,
    decoder2_state: BTreeMap<String, TensorData>,
    decoder1_state_out: BTreeMap<String, TensorData>,
    lm: Vec<f32>,
    query: Vec<f32>,
    query_stop: Vec<f32>,
    done: bool,
    timings: DecoderTimings,
}

impl<'a> EdgeEsrDecoder<'a> {
    pub(crate) fn new(
        decoder1: &'a mut Session,
        decoder2: &'a mut Session,
        attention: MemoryTryAttention,
        active_rows: DecoderActiveRows,
    ) -> Result<Self> {
        let decoder1_state = zero_states(decoder1)?;
        let decoder2_state = zero_states(decoder2)?;
        let mut runtime = Self {
            decoder1,
            decoder2,
            active_rows,
            attention,
            search: OriginalBeamSearch::new(BeamSearchConfig::default())?,
            active_count: 1,
            labels: vec![0; MAX_BEAMS],
            lengths: vec![0; MAX_BEAMS],
            decoder1_state: decoder1_state.clone(),
            decoder2_state,
            decoder1_state_out: decoder1_state,
            lm: vec![0.0; MAX_BEAMS * HIDDEN_SIZE],
            query: vec![0.0; MAX_BEAMS * HIDDEN_SIZE],
            query_stop: vec![0.0; MAX_BEAMS * HIDDEN_SIZE],
            done: false,
            timings: DecoderTimings::default(),
        };
        runtime.labels[0] = START_TOKEN_ID;
        runtime.run_decoder1()?;
        Ok(runtime)
    }

    pub(crate) fn reset(&mut self, attention: MemoryTryAttention) -> Result<()> {
        let decoder1_state = zero_states(self.decoder1)?;
        self.decoder2_state = zero_states(self.decoder2)?;
        self.decoder1_state = decoder1_state.clone();
        self.decoder1_state_out = decoder1_state;
        self.attention = attention;
        self.search = OriginalBeamSearch::new(BeamSearchConfig::default())?;
        self.active_count = 1;
        self.labels.fill(0);
        self.lengths.fill(0);
        self.labels[0] = START_TOKEN_ID;
        self.lm.fill(0.0);
        self.query.fill(0.0);
        self.query_stop.fill(0.0);
        self.done = false;
        self.timings = DecoderTimings::default();
        self.run_decoder1()?;
        Ok(())
    }

    pub(crate) fn advance(
        &mut self,
        bank: &EncoderBank,
        final_flush: bool,
    ) -> Result<DecoderAdvance> {
        if self.done {
            bail!("EdgeEsr decoder is already complete")
        }
        let active_before = self.active_count;
        let attention_started = Instant::now();
        let attention = self.attention.step(MemoryAttentionInput {
            conformer_output: &bank.conformer_output,
            at_h: &bank.at_h,
            chunk_at_h: &bank.chunk_at_h,
            enc_mem_h: &bank.enc_mem_h,
            conformer_mask: &bank.conformer_mask,
            chunk_count: bank.chunk_count,
            frame_width: ENCODER_CHUNK_FRAMES,
            passthrough: &self.lm[..active_before * HIDDEN_SIZE],
            query: &self.query[..active_before * HIDDEN_SIZE],
            query_stop: &self.query_stop[..active_before * HIDDEN_SIZE],
            lengths: &self.lengths[..active_before],
            final_flush,
        })?;
        self.timings.attention += attention_started.elapsed();
        if !attention.ready {
            return Ok(DecoderAdvance { attention });
        }

        let mut context = vec![0.0; MAX_BEAMS * HIDDEN_SIZE];
        let mut lm = vec![0.0; MAX_BEAMS * HIDDEN_SIZE];
        let mut flags = vec![0; MAX_BEAMS];
        context[..active_before * HIDDEN_SIZE].copy_from_slice(&attention.context);
        lm[..active_before * HIDDEN_SIZE].copy_from_slice(&attention.passthrough);
        flags[..active_before].copy_from_slice(&attention.flags);
        let mut inputs = Vec::with_capacity(7);
        inputs.push((
            "context".to_string(),
            Tensor::from_array((vec![MAX_BEAMS, HIDDEN_SIZE], context))?.into_dyn(),
        ));
        inputs.push((
            "lm".to_string(),
            Tensor::from_array((vec![MAX_BEAMS, HIDDEN_SIZE], lm))?.into_dyn(),
        ));
        inputs.push((
            "postproc_flag_in".to_string(),
            Tensor::from_array((vec![MAX_BEAMS], flags))?.into_dyn(),
        ));
        inputs.push((
            "cur_length".to_string(),
            Tensor::from_array((vec![MAX_BEAMS, 1, 1], self.lengths.clone()))?.into_dyn(),
        ));
        for (name, state) in self.decoder2_state.clone() {
            inputs.push((name, state.into_value()?));
        }
        self.active_rows.set(active_before as i32)?;
        let decoder_started = Instant::now();
        let outputs = self.decoder2.run(inputs)?;
        self.timings.onnx += decoder_started.elapsed();
        let (score_shape, scores) = output_f32(&outputs["score"], "score")?;
        if score_shape != [MAX_BEAMS, BeamSearchConfig::default().input_score_count] {
            bail!("decoder part 2 score shape does not match EdgeEsr contract")
        }
        let postproc_flags = output_i32_bits(&outputs["postproc_flag_out"], "postproc_flag_out")?;
        let mut decoder2_state_out = BTreeMap::new();
        for (input_name, output_name) in STATE_NAMES.into_iter().zip(STATE_OUTPUTS) {
            decoder2_state_out.insert(
                input_name.to_string(),
                TensorData::from_value(&outputs[output_name], output_name)?,
            );
        }
        drop(outputs);
        let beam_started = Instant::now();
        let search = self.search.step(
            &scores[..active_before * BeamSearchConfig::default().input_score_count],
            &postproc_flags[..active_before],
            FRAME_BUDGET,
        )?;
        self.timings.beam += beam_started.elapsed();
        let parents = search.parent_indices();
        let active_after = search.candidates.len();
        if active_after == 0 {
            self.active_count = 0;
            self.done = true;
            return Ok(DecoderAdvance { attention });
        }
        self.decoder1_state = gather_states(&self.decoder1_state_out, &parents)?;
        self.decoder2_state = gather_states(&decoder2_state_out, &parents)?;
        self.attention.gather_beams(&parents)?;
        self.active_count = active_after;
        self.labels.fill(0);
        self.labels[..active_after].copy_from_slice(&search.token_ids());
        self.lengths.fill(0);
        self.lengths[..active_after].copy_from_slice(&search.lengths());
        self.run_decoder1()?;
        Ok(DecoderAdvance { attention })
    }

    pub(crate) fn done(&self) -> bool {
        self.done
    }

    pub(crate) fn search(&self) -> &OriginalBeamSearch {
        &self.search
    }

    pub(crate) fn timings(&self) -> DecoderTimings {
        self.timings
    }

    fn run_decoder1(&mut self) -> Result<()> {
        let mut inputs = Vec::with_capacity(5);
        inputs.push((
            "cur_label".to_string(),
            Tensor::from_array((vec![MAX_BEAMS, 1], self.labels.clone()))?.into_dyn(),
        ));
        inputs.push((
            "cur_length".to_string(),
            Tensor::from_array((vec![MAX_BEAMS, 1, 1], self.lengths.clone()))?.into_dyn(),
        ));
        for (name, state) in self.decoder1_state.clone() {
            inputs.push((name, state.into_value()?));
        }
        self.active_rows.set(self.active_count as i32)?;
        let decoder_started = Instant::now();
        let outputs = self.decoder1.run(inputs)?;
        self.timings.onnx += decoder_started.elapsed();
        self.lm = output_f32(&outputs["lm"], "lm")?.1;
        self.query = output_f32(&outputs["lm_att_p2s"], "lm_att_p2s")?.1;
        self.query_stop = output_f32(&outputs["lm_att_p2s_stop"], "lm_att_p2s_stop")?.1;
        if self.lm.len() != MAX_BEAMS * HIDDEN_SIZE
            || self.query.len() != MAX_BEAMS * HIDDEN_SIZE
            || self.query_stop.len() != MAX_BEAMS * HIDDEN_SIZE
        {
            bail!("decoder part 1 output shape does not match EdgeEsr contract")
        }
        self.decoder1_state_out.clear();
        for (input_name, output_name) in STATE_NAMES.into_iter().zip(STATE_OUTPUTS) {
            self.decoder1_state_out.insert(
                input_name.to_string(),
                TensorData::from_value(&outputs[output_name], output_name)?,
            );
        }
        Ok(())
    }
}

fn zero_states(session: &Session) -> Result<BTreeMap<String, TensorData>> {
    STATE_NAMES
        .into_iter()
        .map(|name| {
            Ok((
                name.to_string(),
                TensorData::zeros_for_input(session, name)?,
            ))
        })
        .collect()
}

fn gather_states(
    states: &BTreeMap<String, TensorData>,
    parents: &[usize],
) -> Result<BTreeMap<String, TensorData>> {
    states
        .iter()
        .map(|(name, state)| {
            Ok((
                name.clone(),
                state.gather_rows(parents, MAX_BEAMS)?,
            ))
        })
        .collect()
}
