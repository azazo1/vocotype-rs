#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentReason {
    EndSilence,
    MaxDuration,
    Finish,
}

#[derive(Clone, Debug)]
pub struct SpeechSegment {
    pub samples: Vec<i16>,
    pub reason: SegmentReason,
    pub speech_ms: u32,
    pub start_sample: usize,
    pub end_sample: usize,
    pub audio_start_sample: usize,
    pub audio_end_sample: usize,
}

pub(super) fn expand_bounds(
    speech_start: usize,
    speech_end: usize,
    pre_roll_samples: usize,
    tail_padding_samples: usize,
    available_end: usize,
) -> (usize, usize) {
    let start = speech_start.saturating_sub(pre_roll_samples);
    let end = speech_end
        .saturating_add(tail_padding_samples)
        .min(available_end);
    (start, end)
}
