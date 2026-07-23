use std::path::Path;

use anyhow::Result;
use tracing::{debug, info};

use crate::asr::{AsrEngine, AsrOptions, TranscriptionResult, TARGET_SAMPLE_RATE};
use crate::models::ModelStore;
use crate::punctuation::strip_trailing_period;
use crate::vad::{SpeechSegment, VadConfig, VadSegmenter};

use super::srt::{SubtitleCue, render_srt};
use super::text::{
    default_max_chars, split_text_for_subtitle, split_timed_text_parts,
};

#[derive(Clone, Debug)]
pub struct SubtitleOptions {
    pub max_chars: usize,
    pub asr_options: AsrOptions,
}

impl Default for SubtitleOptions {
    fn default() -> Self {
        Self {
            max_chars: default_max_chars(),
            asr_options: AsrOptions::default(),
        }
    }
}

pub fn transcribe_srt(
    store: ModelStore,
    audio: &Path,
    options: SubtitleOptions,
) -> Result<String> {
    store.verify_required_for(options.asr_options.backend)?;
    if options.asr_options.backend == crate::asr_backend::AsrBackend::Sherpa {
        store.verify_vad_checksum()?;
    }
    let pcm = crate::wav::read_wav_mono_i16(audio, TARGET_SAMPLE_RATE)?;
    let strip_final_period = options.asr_options.strip_trailing_period;
    let mut engine_options = options.asr_options.clone();
    engine_options.strip_trailing_period = false;
    let engine = AsrEngine::load_with_options(store.clone(), engine_options)?;
    info!(
        audio = %audio.display(),
        duration = format!("{:.2}", pcm.duration_seconds()),
        "开始生成 SRT 字幕"
    );

    let mut cues = Vec::new();
    if options.asr_options.backend == crate::asr_backend::AsrBackend::Iflytek {
        let mut segment_count = 0usize;
        engine.transcribe_pcm_streaming(&pcm, |update| {
            let Some(committed) = update.committed_segment else {
                return Ok(());
            };
            segment_count += 1;
            let segment = SpeechSegment {
                samples: committed.samples,
                reason: committed.reason,
                speech_ms: committed.speech_ms,
                start_sample: committed.start_sample,
                end_sample: committed.end_sample,
                audio_start_sample: committed.audio_start_sample,
                audio_end_sample: committed.audio_end_sample,
            };
            push_transcribed_segment(
                &mut cues,
                &segment,
                &committed.result,
                options.max_chars,
            );
            Ok(())
        })?;
        info!(segments = segment_count, "讯飞 VAD 字幕分段完成");
    } else {
        let mut segmenter = VadSegmenter::new_for_backend(
            options.asr_options.backend,
            VadConfig::default(),
            &store.vad_model_path_for(options.asr_options.backend)?,
        )?;
        let segments = segment_audio(&mut segmenter, &pcm.samples)?;
        info!(segments = segments.len(), "sherpa VAD 字幕分段完成");
        for segment in segments {
            let segment_pcm = crate::wav::PcmAudio {
                sample_rate: TARGET_SAMPLE_RATE,
                samples: segment.samples.clone(),
            };
            let result = engine.transcribe_pcm(&segment_pcm)?;
            push_transcribed_segment(&mut cues, &segment, &result, options.max_chars);
        }
    }

    if strip_final_period {
        strip_last_cue_period(&mut cues);
    }

    info!(cues = cues.len(), "SRT 字幕生成完成");
    Ok(render_srt(&cues))
}

fn push_transcribed_segment(
    cues: &mut Vec<SubtitleCue>,
    segment: &SpeechSegment,
    result: &TranscriptionResult,
    max_chars: usize,
) {
    if !result.success || result.text.trim().is_empty() {
        debug!(
            start_ms = sample_to_ms(segment.start_sample),
            end_ms = sample_to_ms(segment.end_sample),
            "跳过空字幕片段"
        );
        return;
    }
    push_segment_cues(
        cues,
        segment,
        &result.text,
        &result.raw_text,
        &result.tokens,
        result.token_timestamps.as_deref(),
        max_chars,
    );
}

fn strip_last_cue_period(cues: &mut [SubtitleCue]) {
    if let Some(last) = cues.last_mut() {
        last.text = strip_trailing_period(&last.text);
    }
}

fn segment_audio(segmenter: &mut VadSegmenter, samples: &[i16]) -> Result<Vec<SpeechSegment>> {
    const FRAME_SAMPLES: usize = 512;

    let mut output = Vec::new();
    for frame in samples.chunks(FRAME_SAMPLES) {
        output.extend(segmenter.push(frame)?);
    }
    output.extend(segmenter.finish()?);
    Ok(output)
}

fn push_segment_cues(
    cues: &mut Vec<SubtitleCue>,
    segment: &SpeechSegment,
    text: &str,
    raw_text: &str,
    tokens: &[String],
    token_timestamps: Option<&[f32]>,
    max_chars: usize,
) {
    if let Some(token_cues) =
        cues_from_token_timestamps(segment, text, raw_text, tokens, token_timestamps, max_chars)
    {
        cues.extend(token_cues);
        return;
    }

    let parts = split_text_for_subtitle(text, max_chars);
    if parts.is_empty() {
        return;
    }

    let start_ms = sample_to_ms(segment.start_sample);
    let end_ms = sample_to_ms(segment.end_sample).max(start_ms + 1);
    let total_chars = parts
        .iter()
        .map(|part| part.chars().count().max(1))
        .sum::<usize>();
    let mut cursor = start_ms;

    for (index, part) in parts.iter().enumerate() {
        let cue_end = if index + 1 == parts.len() {
            end_ms
        } else {
            let chars = part.chars().count().max(1);
            let duration = end_ms.saturating_sub(start_ms);
            let part_ms = duration.saturating_mul(chars as u64) / total_chars as u64;
            (cursor + part_ms.max(500)).min(end_ms)
        };
        cues.push(SubtitleCue {
            start_ms: cursor,
            end_ms: cue_end.max(cursor + 1),
            text: part.clone(),
        });
        cursor = cue_end;
    }
}

fn cues_from_token_timestamps(
    segment: &SpeechSegment,
    text: &str,
    raw_text: &str,
    tokens: &[String],
    token_timestamps: Option<&[f32]>,
    max_chars: usize,
) -> Option<Vec<SubtitleCue>> {
    let timestamps = token_timestamps?;
    if tokens.is_empty() || tokens.len() != timestamps.len() {
        return None;
    }

    let segment_end_seconds = sample_to_seconds(
        segment
            .audio_end_sample
            .saturating_sub(segment.audio_start_sample),
    );
    let timed_groups = split_timed_text_parts(
        text,
        raw_text,
        tokens,
        timestamps,
        max_chars,
        segment_end_seconds,
    );
    if timed_groups.is_empty() {
        return None;
    }

    let offset_ms = sample_to_ms(segment.audio_start_sample);
    let audio_end_ms = sample_to_ms(segment.audio_end_sample).max(offset_ms + 1);

    let mut cues = Vec::new();
    for group in timed_groups {
        let text = group.text.trim();
        if text.is_empty() {
            continue;
        }
        let start_ms = (offset_ms + seconds_to_ms(group.start_seconds)).min(audio_end_ms - 1);
        let end_ms = (offset_ms + seconds_to_ms(group.end_seconds))
            .max(start_ms + 1)
            .min(audio_end_ms);
        cues.push(SubtitleCue {
            start_ms,
            end_ms,
            text: text.to_string(),
        });
    }

    if cues.is_empty() {
        None
    } else {
        Some(cues)
    }
}

fn sample_to_ms(sample: usize) -> u64 {
    (sample as u64).saturating_mul(1_000) / TARGET_SAMPLE_RATE as u64
}

fn sample_to_seconds(sample: usize) -> f32 {
    sample as f32 / TARGET_SAMPLE_RATE as f32
}

fn seconds_to_ms(seconds: f32) -> u64 {
    (seconds.max(0.0) * 1_000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripping_the_document_period_preserves_intermediate_boundaries() {
        let mut cues = vec![
            SubtitleCue {
                start_ms: 0,
                end_ms: 1_000,
                text: "first.".to_string(),
            },
            SubtitleCue {
                start_ms: 1_000,
                end_ms: 2_000,
                text: "second.".to_string(),
            },
        ];

        strip_last_cue_period(&mut cues);

        assert_eq!(cues[0].text, "first.");
        assert_eq!(cues[1].text, "second");
    }

    #[test]
    fn splits_segment_time_by_text_length() {
        let segment = SpeechSegment {
            samples: vec![0; TARGET_SAMPLE_RATE as usize * 2],
            reason: crate::vad::SegmentReason::Finish,
            speech_ms: 2_000,
            start_sample: 0,
            end_sample: TARGET_SAMPLE_RATE as usize * 2,
            audio_start_sample: 0,
            audio_end_sample: TARGET_SAMPLE_RATE as usize * 2,
        };
        let mut cues = Vec::new();
        push_segment_cues(&mut cues, &segment, "你好，世界。", "你好世界", &[], None, 8);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_ms, 0);
        assert_eq!(cues[1].end_ms, 2_000);
    }

    #[test]
    fn cue_time_uses_speech_bounds_not_padded_audio() {
        let segment = SpeechSegment {
            samples: vec![0; TARGET_SAMPLE_RATE as usize * 2],
            reason: crate::vad::SegmentReason::Finish,
            speech_ms: 1_000,
            start_sample: TARGET_SAMPLE_RATE as usize,
            end_sample: TARGET_SAMPLE_RATE as usize * 2,
            audio_start_sample: 0,
            audio_end_sample: TARGET_SAMPLE_RATE as usize * 2,
        };
        let mut cues = Vec::new();
        push_segment_cues(&mut cues, &segment, "测试。", "测试", &[], None, 24);
        assert_eq!(cues[0].start_ms, 1_000);
        assert_eq!(cues[0].end_ms, 2_000);
    }

    #[test]
    fn cue_time_uses_token_timestamps_when_available() {
        let segment = SpeechSegment {
            samples: vec![0; TARGET_SAMPLE_RATE as usize * 2],
            reason: crate::vad::SegmentReason::Finish,
            speech_ms: 2_000,
            start_sample: 0,
            end_sample: TARGET_SAMPLE_RATE as usize * 2,
            audio_start_sample: TARGET_SAMPLE_RATE as usize,
            audio_end_sample: TARGET_SAMPLE_RATE as usize * 3,
        };
        let mut cues = Vec::new();
        let tokens = vec!["你".to_string(), "好".to_string(), "世".to_string(), "界".to_string()];
        let timestamps = vec![0.10, 0.30, 1.10, 1.30];
        push_segment_cues(
            &mut cues,
            &segment,
            "你好，世界。",
            "你好世界",
            &tokens,
            Some(&timestamps),
            2,
        );
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_ms, 1_100);
        assert_eq!(cues[0].end_ms, 2_100);
        assert_eq!(cues[1].end_ms, 2_500);
    }
}
