use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use iflytek_core::MemoryTryAttention;

use crate::decoder::EdgeEsrDecoder;
use crate::encoder::{ENCODER_CHUNK_FRAMES, EdgeEsrEncoder, EncoderChunk};

const HIDDEN_SIZE: usize = 512;
const SEQUENCE_LIMIT: usize = 2_048;
const MAX_DECODER_CALLS: usize = 2_048;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RecognizerTimings {
    pub(crate) encoder: Duration,
    pub(crate) decoder: Duration,
    pub(crate) attention: Duration,
    pub(crate) beam: Duration,
}

#[derive(Default)]
pub(crate) struct EncoderBank {
    pub(crate) conformer_output: Vec<f32>,
    pub(crate) conformer_mask: Vec<f32>,
    pub(crate) at_h: Vec<f32>,
    pub(crate) chunk_at_h: Vec<f32>,
    pub(crate) enc_mem_h: Vec<f32>,
    pub(crate) chunk_count: usize,
}

impl EncoderBank {
    fn append(&mut self, chunk: EncoderChunk) -> Result<()> {
        if self.chunk_count * ENCODER_CHUNK_FRAMES >= SEQUENCE_LIMIT {
            bail!("EdgeEsr encoder exceeded attention sequence capacity")
        }
        let hidden = ENCODER_CHUNK_FRAMES * HIDDEN_SIZE;
        if chunk.conformer_output.len() != hidden
            || chunk.at_h.len() != hidden
            || chunk.chunk_at_h.len() != hidden
            || chunk.enc_mem_h.len() != hidden
            || chunk.conformer_mask.len() != ENCODER_CHUNK_FRAMES
        {
            bail!("EdgeEsr encoder chunk shape changed")
        }
        self.conformer_output.extend(chunk.conformer_output);
        self.conformer_mask.extend(chunk.conformer_mask);
        self.at_h.extend(chunk.at_h);
        self.chunk_at_h.extend(chunk.chunk_at_h);
        self.enc_mem_h.extend(chunk.enc_mem_h);
        self.chunk_count += 1;
        Ok(())
    }
}

pub(crate) struct EdgeEsrRecognizer<'a> {
    encoder: EdgeEsrEncoder<'a>,
    decoder: EdgeEsrDecoder<'a>,
    bank: EncoderBank,
    decoder_calls: usize,
    finalized: bool,
    encoder_elapsed: Duration,
}

impl<'a> EdgeEsrRecognizer<'a> {
    pub(crate) fn new(encoder: EdgeEsrEncoder<'a>, decoder: EdgeEsrDecoder<'a>) -> Self {
        Self {
            encoder,
            decoder,
            bank: EncoderBank::default(),
            decoder_calls: 0,
            finalized: false,
            encoder_elapsed: Duration::ZERO,
        }
    }

    pub(crate) fn reset(&mut self, attention: MemoryTryAttention) -> Result<()> {
        self.encoder.reset()?;
        self.decoder.reset(attention)?;
        self.bank = EncoderBank::default();
        self.decoder_calls = 0;
        self.finalized = false;
        self.encoder_elapsed = Duration::ZERO;
        Ok(())
    }

    pub(crate) fn prime_encoder(&mut self) -> Result<usize> {
        let started = Instant::now();
        let discarded = self.encoder.prime_startup()?;
        self.encoder_elapsed += started.elapsed();
        Ok(discarded.len())
    }

    pub(crate) fn accept_pcm(&mut self, samples: &[i16], final_input: bool) -> Result<()> {
        if self.finalized {
            bail!("EdgeEsr recognizer is already finalized")
        }
        let encoder_started = Instant::now();
        let chunks = self.encoder.accept_pcm(samples, final_input)?;
        self.encoder_elapsed += encoder_started.elapsed();
        let final_group = if final_input { chunks.len().min(2) } else { 0 };
        let normal_end = chunks.len() - final_group;
        for chunk in chunks[..normal_end].iter().cloned() {
            self.bank.append(chunk)?;
            self.pump_decoder(false)?;
        }
        if final_input {
            for chunk in chunks[normal_end..].iter().cloned() {
                self.bank.append(chunk)?;
            }
            self.pump_decoder(true)?;
            self.finalized = true;
        }
        Ok(())
    }

    pub(crate) fn best_path(&self) -> Option<(&[i32], f32)> {
        self.decoder
            .search()
            .best_candidate(self.finalized)
            .map(|candidate| (candidate.path.as_slice(), candidate.normalized_score))
    }

    pub(crate) fn timings(&self) -> RecognizerTimings {
        let decoder = self.decoder.timings();
        RecognizerTimings {
            encoder: self.encoder_elapsed,
            decoder: decoder.onnx,
            attention: decoder.attention,
            beam: decoder.beam,
        }
    }

    fn pump_decoder(&mut self, final_flush: bool) -> Result<()> {
        if self.bank.chunk_count == 0 || self.decoder.done() {
            return Ok(())
        }
        while !self.decoder.done() {
            if self.decoder_calls >= MAX_DECODER_CALLS {
                bail!("EdgeEsr decoder exceeded call safety limit")
            }
            let advance = self.decoder.advance(&self.bank, final_flush)?;
            self.decoder_calls += 1;
            if advance.attention.ready {
                continue;
            }
            if final_flush {
                bail!("EdgeEsr attention stalled during final flush")
            }
            break;
        }
        Ok(())
    }
}
