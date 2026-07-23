use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use tracing::{debug, info, trace};

use crate::recognizer::{EdgeEsrRecognizer, RecognizerTimings};
use crate::token_timing::TokenTimingEstimator;
use crate::{
    EdgeEsrCommittedSegment, EdgeEsrRuntime, EdgeEsrStreamUpdate, EdgeEsrTranscription,
    EdgeEsrVadEvent, EdgeEsrVadSegment, EdgeEsrVadSegmentReason, SAMPLE_RATE,
    STREAM_ACCEPT_SAMPLES, lock_session, samples_to_milliseconds, samples_to_seconds,
};

const FINAL_PADDING_SAMPLES: usize = SAMPLE_RATE * 4 / 5;

#[derive(Default)]
struct CommittedTranscription {
    text: String,
    tokens: Vec<String>,
    token_timestamps: Vec<f32>,
    confidence_sum: f32,
    segment_count: usize,
}

impl CommittedTranscription {
    fn append(&mut self, transcription: &EdgeEsrTranscription, offset_seconds: f32) {
        self.text.push_str(&transcription.text);
        self.tokens.extend(transcription.tokens.iter().cloned());
        if let Some(timestamps) = &transcription.token_timestamps {
            self.token_timestamps
                .extend(timestamps.iter().map(|timestamp| timestamp + offset_seconds));
        } else {
            self.token_timestamps
                .extend(std::iter::repeat_n(offset_seconds, transcription.tokens.len()));
        }
        self.confidence_sum += transcription.confidence;
        self.segment_count += 1;
    }

    fn transcription(&self) -> EdgeEsrTranscription {
        EdgeEsrTranscription {
            text: self.text.clone(),
            tokens: self.tokens.clone(),
            token_timestamps: (!self.tokens.is_empty()
                && self.token_timestamps.len() == self.tokens.len())
                .then(|| self.token_timestamps.clone()),
            confidence: if self.segment_count == 0 {
                0.0
            } else {
                self.confidence_sum / self.segment_count as f32
            },
        }
    }

    fn with_active(
        &self,
        active: &EdgeEsrTranscription,
        offset_seconds: f32,
    ) -> EdgeEsrTranscription {
        let mut output = self.transcription();
        output.text.push_str(&active.text);
        output.tokens.extend(active.tokens.iter().cloned());
        match (&mut output.token_timestamps, &active.token_timestamps) {
            (Some(output), Some(active)) => {
                output.extend(active.iter().map(|timestamp| timestamp + offset_seconds));
            }
            (Some(output), None) => {
                output.extend(std::iter::repeat_n(offset_seconds, active.tokens.len()));
            }
            (None, _) => {}
        }
        if self.segment_count == 0 {
            output.confidence = active.confidence;
        }
        output
    }
}

#[derive(Default)]
struct SessionEmissionState {
    last_text: String,
    revision_count: usize,
    sequence: usize,
    partial_count: usize,
    postprocess_elapsed: Duration,
}

struct EdgeEsrStreamingSession<'runtime, 'sessions, 'emit, Emit>
where
    Emit: FnMut(EdgeEsrStreamUpdate) -> Result<()>,
{
    runtime: &'runtime EdgeEsrRuntime,
    recognizer: EdgeEsrRecognizer<'sessions>,
    emit: &'emit mut Emit,
    audio: Vec<i16>,
    committed: CommittedTranscription,
    emission: SessionEmissionState,
    active: bool,
    recognizer_used: bool,
    asr_cursor: usize,
    active_audio_start: usize,
    active_token_ids: Vec<i32>,
    active_token_timing: TokenTimingEstimator,
    startup_chunks: usize,
    timings: RecognizerTimings,
}

impl<'runtime, 'sessions, 'emit, Emit>
    EdgeEsrStreamingSession<'runtime, 'sessions, 'emit, Emit>
where
    Emit: FnMut(EdgeEsrStreamUpdate) -> Result<()>,
{
    fn new(
        runtime: &'runtime EdgeEsrRuntime,
        recognizer: EdgeEsrRecognizer<'sessions>,
        emit: &'emit mut Emit,
    ) -> Self {
        Self {
            runtime,
            recognizer,
            emit,
            audio: Vec::new(),
            committed: CommittedTranscription::default(),
            emission: SessionEmissionState::default(),
            active: false,
            recognizer_used: false,
            asr_cursor: 0,
            active_audio_start: 0,
            active_token_ids: Vec::new(),
            active_token_timing: TokenTimingEstimator::default(),
            startup_chunks: 0,
            timings: RecognizerTimings::default(),
        }
    }

    fn push_audio(&mut self, samples: &[i16], events: Vec<EdgeEsrVadEvent>) -> Result<()> {
        self.audio.extend_from_slice(samples);
        self.process_events(events)
    }

    fn process_events(&mut self, events: Vec<EdgeEsrVadEvent>) -> Result<()> {
        for event in events {
            match event {
                EdgeEsrVadEvent::SpeechStart { audio_start_sample } => {
                    self.start_segment(audio_start_sample)?;
                }
                EdgeEsrVadEvent::AudioReady {
                    audio_end_sample,
                    force,
                } => self.feed_asr(audio_end_sample, force)?,
                EdgeEsrVadEvent::SpeechEnd(segment) => self.finish_segment(segment)?,
            }
        }
        Ok(())
    }

    fn start_segment(&mut self, audio_start_sample: usize) -> Result<()> {
        if self.active {
            bail!("EdgeEsr VAD started a segment while another segment is active")
        }
        if self.recognizer_used {
            self.recognizer.reset(self.runtime.new_attention()?)?;
        }
        self.startup_chunks += self.recognizer.prime_encoder()?;
        self.recognizer_used = true;
        self.active = true;
        self.active_audio_start = audio_start_sample.min(self.audio.len());
        self.asr_cursor = self.active_audio_start;
        self.active_token_ids.clear();
        self.active_token_timing = TokenTimingEstimator::default();
        debug!(
            audio_start_ms = samples_to_milliseconds(self.active_audio_start),
            segment = self.committed.segment_count,
            "EdgeEsr streaming segment started"
        );
        Ok(())
    }

    fn feed_asr(&mut self, audio_end_sample: usize, force: bool) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let audio_end_sample = audio_end_sample.min(self.audio.len()).max(self.asr_cursor);
        while audio_end_sample.saturating_sub(self.asr_cursor) >= STREAM_ACCEPT_SAMPLES {
            let end = self.asr_cursor + STREAM_ACCEPT_SAMPLES;
            let started = Instant::now();
            self.recognizer
                .accept_pcm(&self.audio[self.asr_cursor..end], false)?;
            self.asr_cursor = end;
            trace!(
                elapsed_us = started.elapsed().as_micros(),
                accepted_samples = STREAM_ACCEPT_SAMPLES,
                "EdgeEsr VAD streaming chunk accepted"
            );
            self.emit_active_partial()?;
        }
        if force && self.asr_cursor < audio_end_sample {
            let start = self.asr_cursor;
            self.recognizer
                .accept_pcm(&self.audio[start..audio_end_sample], false)?;
            self.asr_cursor = audio_end_sample;
            self.emit_active_partial()?;
        }
        Ok(())
    }

    fn emit_active_partial(&mut self) -> Result<()> {
        let Some((path, _)) = self.recognizer.best_path() else {
            return Ok(());
        };
        if path == self.active_token_ids.as_slice() {
            return Ok(());
        }
        self.active_token_ids.clear();
        self.active_token_ids.extend_from_slice(path);
        let started = Instant::now();
        let mut active = self.runtime.recognizer_transcription(&self.recognizer, false)?;
        self.emission.postprocess_elapsed += started.elapsed();
        let active_samples = self.asr_cursor.saturating_sub(self.active_audio_start);
        active.token_timestamps = self
            .active_token_timing
            .update(&active.tokens, samples_to_seconds(active_samples));
        if active.text.is_empty() {
            return Ok(());
        }
        let transcription = self.committed.with_active(
            &active,
            samples_to_seconds(self.active_audio_start),
        );
        self.emission.partial_count += 1;
        self.emit_update(transcription, None, false)
    }

    fn finish_segment(&mut self, segment: EdgeEsrVadSegment) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.feed_asr(segment.audio_end_sample, true)?;
        self.recognizer
            .accept_pcm(&vec![0; FINAL_PADDING_SAMPLES], true)?;
        let segment_timings = self.recognizer.timings();
        self.timings.encoder += segment_timings.encoder;
        self.timings.decoder += segment_timings.decoder;
        self.timings.attention += segment_timings.attention;
        self.timings.beam += segment_timings.beam;

        let started = Instant::now();
        let mut transcription = self.runtime.recognizer_transcription(&self.recognizer, true)?;
        self.emission.postprocess_elapsed += started.elapsed();
        let active_samples = self.asr_cursor.saturating_sub(self.active_audio_start);
        transcription.token_timestamps = self
            .active_token_timing
            .update(&transcription.tokens, samples_to_seconds(active_samples));
        let committed_audio_end = self
            .asr_cursor
            .max(segment.audio_end_sample)
            .min(self.audio.len());
        let committed_segment = EdgeEsrCommittedSegment {
            transcription: transcription.clone(),
            samples: self.audio
                [segment.audio_start_sample.min(self.audio.len())
                    ..committed_audio_end]
                .to_vec(),
            reason: segment.reason.clone(),
            speech_ms: segment.speech_ms,
            start_sample: segment.start_sample,
            end_sample: segment.end_sample,
            audio_start_sample: segment.audio_start_sample,
            audio_end_sample: committed_audio_end,
        };
        self.committed.append(
            &transcription,
            samples_to_seconds(segment.audio_start_sample),
        );
        self.active = false;
        self.active_token_ids.clear();
        self.active_token_timing = TokenTimingEstimator::default();
        self.asr_cursor = segment.audio_end_sample.min(self.audio.len());
        debug!(
            segment = self.committed.segment_count,
            speech_ms = segment.speech_ms,
            recognized_audio_end_ms = samples_to_milliseconds(committed_audio_end),
            committed_chars = self.committed.text.chars().count(),
            "EdgeEsr streaming segment committed"
        );
        self.emit_update(
            self.committed.transcription(),
            Some(committed_segment),
            false,
        )
    }

    fn finish(mut self) -> Result<(EdgeEsrTranscription, SessionEmissionState, RecognizerTimings, usize, usize)> {
        if self.active {
            let start = self.active_audio_start.min(self.audio.len());
            let end = self.audio.len();
            self.finish_segment(EdgeEsrVadSegment {
                samples: self.audio[start..end].to_vec(),
                reason: EdgeEsrVadSegmentReason::Finish,
                speech_ms: ((end.saturating_sub(start) as u64 * 1_000) / SAMPLE_RATE as u64)
                    as u32,
                start_sample: start,
                end_sample: end,
                audio_start_sample: start,
                audio_end_sample: end,
            })?;
        }
        let transcription = self.committed.transcription();
        self.emit_update(transcription.clone(), None, true)?;
        Ok((
            transcription,
            self.emission,
            self.timings,
            self.startup_chunks,
            self.committed.segment_count,
        ))
    }

    fn emit_update(
        &mut self,
        transcription: EdgeEsrTranscription,
        committed_segment: Option<EdgeEsrCommittedSegment>,
        final_result: bool,
    ) -> Result<()> {
        if !final_result
            && committed_segment.is_none()
            && transcription.text == self.emission.last_text
        {
            return Ok(());
        }
        let revision = !self.emission.last_text.is_empty()
            && !transcription.text.starts_with(&self.emission.last_text);
        if revision {
            self.emission.revision_count += 1;
        }
        self.emission.sequence += 1;
        self.emission.last_text = transcription.text.clone();
        (self.emit)(EdgeEsrStreamUpdate {
            committed_prefix_chars: self.committed.text.chars().count(),
            transcription,
            committed_segment,
            revision,
            revision_count: self.emission.revision_count,
            sequence: self.emission.sequence,
            final_result,
        })
    }
}

impl EdgeEsrRuntime {
    pub fn transcribe_vad_streaming_pcm<Read, Emit>(
        &self,
        sample_rate: u32,
        mut read: Read,
        mut emit: Emit,
    ) -> Result<EdgeEsrTranscription>
    where
        Read: FnMut(&mut Vec<i16>) -> Result<bool>,
        Emit: FnMut(EdgeEsrStreamUpdate) -> Result<()>,
    {
        if sample_rate as usize != SAMPLE_RATE {
            bail!("EdgeEsr ASR requires a 16000 Hz sample rate")
        }
        let started = Instant::now();
        let _decoder_guard = self.decoder_guard()?;
        let mut vad = self
            .sessions
            .vad
            .lock()
            .map_err(|_| anyhow::anyhow!("EdgeEsr VAD lock poisoned"))?;
        let mut vgg = lock_session(&self.sessions.vgg_encoder)?;
        let mut conformer = lock_session(&self.sessions.conformer_encoder)?;
        let mut decoder1 = lock_session(&self.sessions.decoder_part1)?;
        let mut decoder2 = lock_session(&self.sessions.decoder_part2)?;
        let recognizer = self.new_recognizer(
            &mut vgg,
            &mut conformer,
            &mut decoder1,
            &mut decoder2,
        )?;
        vad.reset();
        let result = (|| {
            let mut session = EdgeEsrStreamingSession::new(self, recognizer, &mut emit);
            loop {
                let mut incoming = Vec::new();
                let has_more = read(&mut incoming)?;
                let events = vad.push_events(&incoming)?;
                session.push_audio(&incoming, events)?;
                if !has_more {
                    break;
                }
            }
            let events = vad.finish_events()?;
            session.process_events(events)?;
            let input_samples = session.audio.len();
            let (transcription, emission, timings, startup_chunks, segment_count) = session.finish()?;
            let compute_elapsed = timings.encoder
                + timings.decoder
                + timings.attention
                + timings.beam
                + emission.postprocess_elapsed;
            info!(
                stream_elapsed_ms = started.elapsed().as_millis(),
                compute_ms = compute_elapsed.as_millis(),
                encoder_ms = timings.encoder.as_millis(),
                decoder_ms = timings.decoder.as_millis(),
                attention_ms = timings.attention.as_millis(),
                beam_ms = timings.beam.as_millis(),
                postprocess_ms = emission.postprocess_elapsed.as_millis(),
                audio_ms = samples_to_milliseconds(input_samples),
                startup_chunks,
                segment_count,
                partial_count = emission.partial_count,
                revision_count = emission.revision_count,
                token_count = transcription.tokens.len(),
                "EdgeEsr VAD streaming transcription completed"
            );
            Ok(transcription)
        })();
        vad.reset();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcription(text: &str, confidence: f32) -> EdgeEsrTranscription {
        EdgeEsrTranscription {
            text: text.to_string(),
            tokens: text.chars().map(|character| character.to_string()).collect(),
            token_timestamps: Some(vec![0.0; text.chars().count()]),
            confidence,
        }
    }

    #[test]
    fn committed_segments_preserve_their_boundary_text() {
        let mut committed = CommittedTranscription::default();
        committed.append(&transcription("first.", 0.8), 0.0);
        committed.append(&transcription("second.", 0.6), 2.0);

        let result = committed.transcription();
        assert_eq!(result.text, "first.second.");
        assert_eq!(result.tokens.len(), result.token_timestamps.unwrap().len());
        assert!((result.confidence - 0.7).abs() < 0.0001);
    }

    #[test]
    fn active_text_is_appended_after_the_committed_prefix() {
        let mut committed = CommittedTranscription::default();
        committed.append(&transcription("stable,", 0.9), 0.0);

        let result = committed.with_active(&transcription("partial", 0.5), 1.0);
        assert_eq!(result.text, "stable,partial");
        assert_eq!(result.tokens.len(), result.token_timestamps.unwrap().len());
    }
}
