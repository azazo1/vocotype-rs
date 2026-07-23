use std::path::Path;

use anyhow::Result;

use crate::asr_backend::AsrBackend;

use super::config::VadConfig;
use super::segment::{SegmentReason, SpeechSegment};
use super::segmenter::SherpaVadSegmenter;

pub enum VadSegmenter {
    Sherpa(SherpaVadSegmenter),
    Iflytek(Box<iflytek_runtime::EdgeEsrVad>),
}

impl VadSegmenter {
    pub fn new(config: VadConfig, model_path: &Path) -> Result<Self> {
        Ok(Self::Sherpa(SherpaVadSegmenter::new(
            config,
            model_path,
        )?))
    }

    pub fn new_for_backend(
        backend: AsrBackend,
        config: VadConfig,
        model_path: &Path,
    ) -> Result<Self> {
        match backend {
            AsrBackend::Sherpa => Self::new(config, model_path),
            AsrBackend::Iflytek => Ok(Self::Iflytek(Box::new(
                iflytek_runtime::EdgeEsrVad::load(
                    model_path,
                    iflytek_runtime::EdgeEsrVadConfig {
                        sample_rate: config.sample_rate,
                        threshold: config.threshold,
                        pre_roll_ms: config.pre_roll_ms,
                        tail_padding_ms: config.tail_padding_ms,
                        end_silence_ms: config.end_silence_ms,
                        min_speech_ms: config.min_speech_ms,
                        max_segment_ms: config.max_segment_ms,
                    },
                )?,
            ))),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Sherpa(segmenter) => segmenter.reset(),
            Self::Iflytek(segmenter) => segmenter.reset(),
        }
    }

    pub fn push(&mut self, frame: &[i16]) -> Result<Vec<SpeechSegment>> {
        match self {
            Self::Sherpa(segmenter) => Ok(segmenter.push(frame)),
            Self::Iflytek(segmenter) => segmenter
                .push(frame)
                .map(|segments| segments.into_iter().map(map_iflytek_segment).collect()),
        }
    }

    pub fn finish(&mut self) -> Result<Vec<SpeechSegment>> {
        match self {
            Self::Sherpa(segmenter) => Ok(segmenter.finish()),
            Self::Iflytek(segmenter) => segmenter
                .finish()
                .map(|segments| segments.into_iter().map(map_iflytek_segment).collect()),
        }
    }

    pub fn detected(&self) -> bool {
        match self {
            Self::Sherpa(segmenter) => segmenter.detected(),
            Self::Iflytek(segmenter) => segmenter.detected(),
        }
    }
}

fn map_iflytek_segment(segment: iflytek_runtime::EdgeEsrVadSegment) -> SpeechSegment {
    let reason = match segment.reason {
        iflytek_runtime::EdgeEsrVadSegmentReason::EndSilence => SegmentReason::EndSilence,
        iflytek_runtime::EdgeEsrVadSegmentReason::MaxDuration => SegmentReason::MaxDuration,
        iflytek_runtime::EdgeEsrVadSegmentReason::Finish => SegmentReason::Finish,
    };
    SpeechSegment {
        samples: segment.samples,
        reason,
        speech_ms: segment.speech_ms,
        start_sample: segment.start_sample,
        end_sample: segment.end_sample,
        audio_start_sample: segment.audio_start_sample,
        audio_end_sample: segment.audio_end_sample,
    }
}
