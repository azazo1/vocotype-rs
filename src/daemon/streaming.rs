use std::collections::VecDeque;

use crate::asr::TranscriptionUpdate;

const STABILITY_HISTORY: usize = 5;
const UNSTABLE_TAIL_TOKENS: usize = 8;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct StreamPresentation {
    pub(super) stable: String,
    pub(super) unstable: String,
    pub(super) revision: bool,
}

#[derive(Clone, Debug)]
struct StreamSnapshot {
    text: String,
    tokens: Vec<String>,
}

#[derive(Default)]
pub(super) struct StablePrefixTracker {
    history: VecDeque<StreamSnapshot>,
    revision_count: usize,
}

impl StablePrefixTracker {
    pub(super) fn update(&mut self, update: &TranscriptionUpdate) -> StreamPresentation {
        if update.final_result {
            return StreamPresentation {
                stable: update.result.text.clone(),
                unstable: String::new(),
                revision: update.revision,
            };
        }

        let current = &update.result.text;
        let committed = current
            .chars()
            .take(update.committed_prefix_chars)
            .collect::<String>();
        if update.revision || update.revision_count > self.revision_count {
            self.history.clear();
        }
        self.revision_count = update.revision_count;

        self.history.push_back(StreamSnapshot {
            text: update.result.text.clone(),
            tokens: update.result.tokens.clone(),
        });
        while self.history.len() > STABILITY_HISTORY {
            let _ = self.history.pop_front();
        }

        if self.history.len() < STABILITY_HISTORY {
            return self.presentation(current, committed, update.revision);
        }

        let common_tokens = common_token_prefix(&self.history);
        let tail_limit = update
            .result
            .tokens
            .len()
            .saturating_sub(UNSTABLE_TAIL_TOKENS);
        let stable_tokens = common_tokens.min(tail_limit);

        let token_prefix = update.result.tokens[..stable_tokens].concat();
        let text_prefix_chars = common_text_prefix_chars(&self.history);
        let candidate_chars = token_prefix.chars().count().min(text_prefix_chars);
        let candidate = current.chars().take(candidate_chars).collect::<String>();
        let inferred = candidate[..last_stable_boundary(&candidate)].to_string();
        let stable = if inferred.chars().count() >= committed.chars().count() {
            inferred
        } else {
            committed
        };
        self.presentation(current, stable, update.revision)
    }

    fn presentation(
        &self,
        current: &str,
        stable: String,
        revision: bool,
    ) -> StreamPresentation {
        let (stable, unstable) = if let Some(unstable) = current.strip_prefix(&stable) {
            (stable, unstable.to_string())
        } else {
            (String::new(), current.to_string())
        };
        StreamPresentation {
            stable,
            unstable,
            revision,
        }
    }
}

fn common_token_prefix(history: &VecDeque<StreamSnapshot>) -> usize {
    let Some(first) = history.front() else {
        return 0;
    };
    let mut length = first.tokens.len();
    for snapshot in history.iter().skip(1) {
        length = length.min(snapshot.tokens.len());
        while length > 0 && first.tokens[..length] != snapshot.tokens[..length] {
            length -= 1;
        }
    }
    length
}

fn common_text_prefix_chars(history: &VecDeque<StreamSnapshot>) -> usize {
    let Some(first) = history.front() else {
        return 0;
    };
    let first = first.text.chars().collect::<Vec<_>>();
    let mut length = first.len();
    for snapshot in history.iter().skip(1) {
        let text = snapshot.text.chars().collect::<Vec<_>>();
        length = length.min(text.len());
        while length > 0 && first[..length] != text[..length] {
            length -= 1;
        }
    }
    length
}

fn last_stable_boundary(text: &str) -> usize {
    let characters = text.char_indices().collect::<Vec<_>>();
    let mut boundary = 0;
    for (index, (byte_index, character)) in characters.iter().copied().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|index| characters.get(index))
            .map(|(_, character)| *character);
        let next = characters
            .get(index + 1)
            .map(|(_, character)| *character);
        if is_stable_boundary(character, previous, next) {
            boundary = byte_index + character.len_utf8();
        }
    }
    boundary
}

fn is_stable_boundary(
    character: char,
    previous: Option<char>,
    next: Option<char>,
) -> bool {
    if matches!(
        character,
        '\u{ff0c}' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}' | '\u{ff1b}'
    ) {
        return true;
    }
    if !matches!(character, ',' | '.' | '!' | '?' | ';') {
        return false;
    }
    !matches!(
        (previous, next),
        (Some(previous), Some(next))
            if previous.is_ascii_alphanumeric() && next.is_ascii_alphanumeric()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::TranscriptionResult;

    fn update_with_revision(
        text: &str,
        revision: bool,
        revision_count: usize,
        final_result: bool,
    ) -> TranscriptionUpdate {
        TranscriptionUpdate {
            result: TranscriptionResult {
                success: true,
                text: text.to_string(),
                raw_text: text.to_string(),
                tokens: text.chars().map(|character| character.to_string()).collect(),
                token_timestamps: None,
                duration: 1.0,
                inference_latency: 0.1,
                confidence: 1.0,
                error: None,
            },
            committed_prefix_chars: 0,
            committed_segment: None,
            revision,
            revision_count,
            sequence: 1,
            final_result,
        }
    }

    fn update(text: &str, final_result: bool) -> TranscriptionUpdate {
        update_with_revision(text, false, 0, final_result)
    }

    fn stabilize_first_clause(tracker: &mut StablePrefixTracker) -> StreamPresentation {
        let _ = tracker.update(&update("这是一个离线语音识别模型测试，今天", false));
        let _ = tracker.update(&update("这是一个离线语音识别模型测试，今天天气", false));
        let _ = tracker.update(&update("这是一个离线语音识别模型测试，今天天气很好", false));
        let _ = tracker.update(&update(
            "这是一个离线语音识别模型测试，今天天气很好我们",
            false,
        ));
        tracker.update(&update(
            "这是一个离线语音识别模型测试，今天天气很好我们正在",
            false,
        ))
    }

    #[test]
    fn marks_only_a_punctuation_terminated_clause_as_stable() {
        let mut tracker = StablePrefixTracker::default();
        let result = stabilize_first_clause(&mut tracker);
        assert_eq!(result.stable, "这是一个离线语音识别模型测试，");
        assert_eq!(result.unstable, "今天天气很好我们正在");
    }

    #[test]
    fn keeps_a_partial_clause_without_punctuation_unstable() {
        let mut tracker = StablePrefixTracker::default();
        for _ in 0..STABILITY_HISTORY {
            let result = tracker.update(&update("这是一个仍会继续校准的长句子", false));
            assert!(result.stable.is_empty());
        }
    }

    #[test]
    fn exposes_runtime_committed_prefix_immediately() {
        let mut tracker = StablePrefixTracker::default();
        let mut partial = update("stable,changing", false);
        partial.committed_prefix_chars = "stable,".chars().count();

        let result = tracker.update(&partial);

        assert_eq!(result.stable, "stable,");
        assert_eq!(result.unstable, "changing");
    }

    #[test]
    fn final_result_is_entirely_stable() {
        let mut tracker = StablePrefixTracker::default();
        let _ = stabilize_first_clause(&mut tracker);
        let final_result = tracker.update(&update(
            "这是一个离线语音识别模型测试，今天天气很好。",
            true,
        ));
        assert_eq!(
            final_result.stable,
            "这是一个离线语音识别模型测试，今天天气很好。"
        );
        assert!(final_result.unstable.is_empty());
    }

    #[test]
    fn does_not_treat_decimal_punctuation_as_a_stable_boundary() {
        let mut tracker = StablePrefixTracker::default();
        for _ in 0..STABILITY_HISTORY {
            let partial = tracker.update(&update("版本12.34完成后继续处理", false));
            assert!(partial.stable.is_empty());
        }
    }

    #[test]
    fn revision_resets_stability_and_can_stabilize_again() {
        let mut tracker = StablePrefixTracker::default();
        let _ = stabilize_first_clause(&mut tracker);
        let revision = tracker.update(&update_with_revision(
            "那是一个离线语音识别模型测试，今天天气真好",
            true,
            1,
            false,
        ));
        assert!(revision.stable.is_empty());

        let mut partial = revision;
        for _ in 1..STABILITY_HISTORY {
            partial = tracker.update(&update_with_revision(
                "那是一个离线语音识别模型测试，今天天气真好，我们正在继续",
                false,
                1,
                false,
            ));
        }
        assert_eq!(partial.stable, "那是一个离线语音识别模型测试，");
    }
}
