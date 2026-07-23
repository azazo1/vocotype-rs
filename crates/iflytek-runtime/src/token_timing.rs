#[derive(Clone, Debug)]
struct TokenSnapshot {
    tokens: Vec<String>,
    audio_end_seconds: f32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TokenTimingEstimator {
    snapshots: Vec<TokenSnapshot>,
}

impl TokenTimingEstimator {
    pub(crate) fn update(
        &mut self,
        tokens: &[String],
        audio_end_seconds: f32,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() {
            return None;
        }

        let audio_end_seconds = self.normalized_audio_end(audio_end_seconds);
        if self
            .snapshots
            .last()
            .is_none_or(|snapshot| snapshot.tokens != tokens)
        {
            self.snapshots.push(TokenSnapshot {
                tokens: tokens.to_vec(),
                audio_end_seconds,
            });
        }

        Some(self.estimate(tokens, audio_end_seconds))
    }

    fn estimate(&self, tokens: &[String], audio_end_seconds: f32) -> Vec<f32> {
        let mut deadlines = vec![None; tokens.len()];
        for snapshot in &self.snapshots {
            let prefix_len = common_prefix_len(&snapshot.tokens, tokens);
            for deadline in deadlines.iter_mut().take(prefix_len) {
                if deadline.is_none() {
                    *deadline = Some(snapshot.audio_end_seconds.min(audio_end_seconds));
                }
            }
        }

        let mut previous_deadline = 0.0_f32;
        let deadlines = deadlines
            .into_iter()
            .map(|deadline| {
                let deadline = deadline
                    .unwrap_or(audio_end_seconds)
                    .clamp(previous_deadline, audio_end_seconds);
                previous_deadline = deadline;
                deadline
            })
            .collect::<Vec<_>>();

        let mut timestamps = vec![0.0_f32; tokens.len()];
        let mut interval_start = 0.0_f32;
        let mut group_start = 0_usize;
        while group_start < deadlines.len() {
            let interval_end = deadlines[group_start].max(interval_start);
            let mut group_end = group_start + 1;
            while group_end < deadlines.len() && deadlines[group_end] == deadlines[group_start] {
                group_end += 1;
            }

            let group_len = group_end - group_start;
            let interval = interval_end - interval_start;
            for offset in 0..group_len {
                timestamps[group_start + offset] =
                    interval_start + interval * offset as f32 / group_len as f32;
            }
            interval_start = interval_end;
            group_start = group_end;
        }

        let mut previous_timestamp = 0.0_f32;
        for timestamp in &mut timestamps {
            *timestamp = timestamp.clamp(previous_timestamp, audio_end_seconds);
            previous_timestamp = *timestamp;
        }
        timestamps
    }

    fn normalized_audio_end(&self, audio_end_seconds: f32) -> f32 {
        let audio_end_seconds = if audio_end_seconds.is_finite() {
            audio_end_seconds.max(0.0)
        } else {
            0.0
        };
        self.snapshots
            .last()
            .map_or(audio_end_seconds, |snapshot| {
                audio_end_seconds.max(snapshot.audio_end_seconds)
            })
    }
}

fn common_prefix_len(left: &[String], right: &[String]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn assert_timestamps(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.0001, "{actual} != {expected}");
        }
    }

    #[test]
    fn growing_partials_produce_increasing_timestamps() {
        let mut estimator = TokenTimingEstimator::default();
        let _ = estimator.update(&tokens(&["a"]), 0.64);
        let _ = estimator.update(&tokens(&["a", "b"]), 1.28);
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 1.92)
            .unwrap();

        assert_timestamps(&timestamps, &[0.0, 0.64, 1.28]);
    }

    #[test]
    fn tokens_from_one_partial_share_its_audio_interval() {
        let mut estimator = TokenTimingEstimator::default();
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 0.96)
            .unwrap();

        assert_timestamps(&timestamps, &[0.0, 0.32, 0.64]);
    }

    #[test]
    fn revised_suffix_does_not_affect_final_timestamps() {
        let mut estimator = TokenTimingEstimator::default();
        let _ = estimator.update(&tokens(&["a", "wrong"]), 0.64);
        let _ = estimator.update(&tokens(&["a", "other"]), 1.28);
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 1.92)
            .unwrap();

        assert_timestamps(&timestamps, &[0.0, 0.64, 1.28]);
    }

    #[test]
    fn final_only_suffix_fills_the_remaining_interval() {
        let mut estimator = TokenTimingEstimator::default();
        let _ = estimator.update(&tokens(&["a"]), 0.64);
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 2.0)
            .unwrap();

        assert_timestamps(&timestamps, &[0.0, 0.64, 1.32]);
    }

    #[test]
    fn final_only_transcription_spans_the_audio_duration() {
        let mut estimator = TokenTimingEstimator::default();
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 2.0)
            .unwrap();

        assert_timestamps(&timestamps, &[0.0, 2.0 / 3.0, 4.0 / 3.0]);
    }

    #[test]
    fn empty_tokens_do_not_produce_timestamps() {
        let mut estimator = TokenTimingEstimator::default();

        assert!(estimator.update(&[], 1.0).is_none());
    }

    #[test]
    fn timestamps_stay_monotonic_and_inside_audio_bounds() {
        let mut estimator = TokenTimingEstimator::default();
        let _ = estimator.update(&tokens(&["a"]), 1.5);
        let timestamps = estimator
            .update(&tokens(&["a", "b", "c"]), 1.0)
            .unwrap();

        assert_eq!(timestamps.len(), 3);
        assert!(timestamps.windows(2).all(|window| window[0] <= window[1]));
        assert!(timestamps.iter().all(|timestamp| (0.0..=1.5).contains(timestamp)));
    }
}
