use std::collections::VecDeque;

use crate::asr::TranscriptionUpdate;

const STABILITY_HISTORY: usize = 3;
const UNSTABLE_TAIL_TOKENS: usize = 3;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct StreamPresentation {
    pub(super) stable: String,
    pub(super) unstable: String,
    pub(super) commit: String,
    pub(super) revision: bool,
    pub(super) conflict: bool,
    pub(super) final_result: bool,
}

#[derive(Clone, Debug)]
struct StreamSnapshot {
    text: String,
    tokens: Vec<String>,
}

#[derive(Default)]
pub(super) struct StablePrefixTracker {
    history: VecDeque<StreamSnapshot>,
    committed: String,
}

impl StablePrefixTracker {
    pub(super) fn update(&mut self, update: &TranscriptionUpdate) -> StreamPresentation {
        if update.final_result {
            return self.finish(update);
        }

        self.history.push_back(StreamSnapshot {
            text: update.result.text.clone(),
            tokens: update.result.tokens.clone(),
        });
        while self.history.len() > STABILITY_HISTORY {
            let _ = self.history.pop_front();
        }

        let current = &update.result.text;
        if self.history.len() < STABILITY_HISTORY {
            return self.presentation(current, String::new(), update.revision, false, false);
        }

        let common_tokens = common_token_prefix(&self.history);
        let tail_limit = update
            .result
            .tokens
            .len()
            .saturating_sub(UNSTABLE_TAIL_TOKENS);
        let mut stable_tokens = common_tokens.min(tail_limit);
        while stable_tokens > 0
            && token_needs_final_boundary(&update.result.tokens[stable_tokens - 1])
        {
            stable_tokens -= 1;
        }

        let token_prefix = update.result.tokens[..stable_tokens].concat();
        let text_prefix_chars = common_text_prefix_chars(&self.history);
        let candidate_chars = token_prefix.chars().count().min(text_prefix_chars);
        let candidate = current.chars().take(candidate_chars).collect::<String>();
        if !candidate.starts_with(&self.committed) {
            return self.presentation(current, String::new(), update.revision, true, false);
        }

        let commit = candidate[self.committed.len()..].to_string();
        self.committed = candidate;
        self.presentation(current, commit, update.revision, false, false)
    }

    fn finish(&mut self, update: &TranscriptionUpdate) -> StreamPresentation {
        let current = &update.result.text;
        if !current.starts_with(&self.committed) {
            return self.presentation(current, String::new(), update.revision, true, true);
        }
        let commit = current[self.committed.len()..].to_string();
        self.committed = current.clone();
        self.presentation(current, commit, update.revision, false, true)
    }

    fn presentation(
        &self,
        current: &str,
        commit: String,
        revision: bool,
        conflict: bool,
        final_result: bool,
    ) -> StreamPresentation {
        let (stable, unstable) = if let Some(unstable) = current.strip_prefix(&self.committed) {
            (self.committed.clone(), unstable.to_string())
        } else {
            (String::new(), current.to_string())
        };
        StreamPresentation {
            stable,
            unstable,
            commit,
            revision,
            conflict,
            final_result,
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

fn token_needs_final_boundary(token: &str) -> bool {
    token.chars().any(|character| {
        character.is_ascii_alphanumeric()
            || matches!(
                character,
                ',' | '.' | ':' | ';' | '!' | '?' | '%' | '/' | '-' | '+' | '=' | '@' | '#'
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::TranscriptionResult;

    fn update(text: &str, final_result: bool) -> TranscriptionUpdate {
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
            revision: false,
            revision_count: 0,
            sequence: 1,
            final_result,
        }
    }

    #[test]
    fn commits_only_the_prefix_shared_by_three_partials() {
        let mut tracker = StablePrefixTracker::default();
        assert!(tracker.update(&update("这是一个无线", false)).commit.is_empty());
        assert!(
            tracker
                .update(&update("这是一个离线语音", false))
                .commit
                .is_empty()
        );
        let result = tracker.update(&update("这是一个离线语音识别", false));
        assert_eq!(result.commit, "这是一个");
        assert_eq!(result.stable, "这是一个");
        assert_eq!(result.unstable, "离线语音识别");
    }

    #[test]
    fn final_commits_only_the_remaining_suffix() {
        let mut tracker = StablePrefixTracker::default();
        let _ = tracker.update(&update("这是一个离线语音", false));
        let _ = tracker.update(&update("这是一个离线语音识别", false));
        let partial = tracker.update(&update("这是一个离线语音识别模型", false));
        assert_eq!(partial.commit, "这是一个离线语音");
        let final_result = tracker.update(&update("这是一个离线语音识别模型测试,", true));
        assert_eq!(final_result.commit, "识别模型测试,");
        assert!(final_result.unstable.is_empty());
    }

    #[test]
    fn keeps_ascii_and_punctuation_at_the_unstable_boundary() {
        let mut tracker = StablePrefixTracker::default();
        let _ = tracker.update(&update("版本12.", false));
        let _ = tracker.update(&update("版本12.3", false));
        let partial = tracker.update(&update("版本12.34完成", false));
        assert_eq!(partial.commit, "版本");
        assert_eq!(partial.unstable, "12.34完成");
    }

    #[test]
    fn reports_conflict_when_revision_crosses_the_committed_prefix() {
        let mut tracker = StablePrefixTracker::default();
        let _ = tracker.update(&update("这是一个离线语音", false));
        let _ = tracker.update(&update("这是一个离线语音", false));
        let committed = tracker.update(&update("这是一个离线语音", false));
        assert!(!committed.commit.is_empty());

        let revision = tracker.update(&update("那是一个离线语音", false));
        assert!(revision.conflict);
        assert!(revision.commit.is_empty());
    }
}
