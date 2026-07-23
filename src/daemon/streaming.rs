use std::collections::VecDeque;

use crate::asr::TranscriptionUpdate;

const STABILITY_HISTORY: usize = 5;
const UNSTABLE_TAIL_TOKENS: usize = 8;

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
    revision_seen: bool,
}

impl StablePrefixTracker {
    pub(super) fn update(&mut self, update: &TranscriptionUpdate) -> StreamPresentation {
        if update.final_result {
            return self.finish(update);
        }

        let current = &update.result.text;
        if !current.starts_with(&self.committed) {
            return self.presentation(current, String::new(), update.revision, true, false);
        }

        if update.revision || (update.revision_count > 0 && !self.revision_seen) {
            self.revision_seen = true;
            self.history.clear();
        }

        self.history.push_back(StreamSnapshot {
            text: update.result.text.clone(),
            tokens: update.result.tokens.clone(),
        });
        while self.history.len() > STABILITY_HISTORY {
            let _ = self.history.pop_front();
        }

        if self.revision_seen || self.history.len() < STABILITY_HISTORY {
            return self.presentation(current, String::new(), update.revision, false, false);
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
        let candidate = candidate[..last_commit_boundary(&candidate)].to_string();
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

fn last_commit_boundary(text: &str) -> usize {
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
        if is_commit_boundary(character, previous, next) {
            boundary = byte_index + character.len_utf8();
        }
    }
    boundary
}

fn is_commit_boundary(
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
            revision,
            revision_count,
            sequence: 1,
            final_result,
        }
    }

    fn update(text: &str, final_result: bool) -> TranscriptionUpdate {
        update_with_revision(text, false, 0, final_result)
    }

    fn commit_first_clause(tracker: &mut StablePrefixTracker) -> StreamPresentation {
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
    fn commits_only_a_punctuation_terminated_stable_clause() {
        let mut tracker = StablePrefixTracker::default();
        let result = commit_first_clause(&mut tracker);
        assert_eq!(result.commit, "这是一个离线语音识别模型测试，");
        assert_eq!(result.stable, "这是一个离线语音识别模型测试，");
        assert_eq!(result.unstable, "今天天气很好我们正在");
    }

    #[test]
    fn does_not_commit_a_partial_clause_without_punctuation() {
        let mut tracker = StablePrefixTracker::default();
        for _ in 0..STABILITY_HISTORY {
            let result = tracker.update(&update("这是一个仍会继续校准的长句子", false));
            assert!(result.commit.is_empty());
        }
    }

    #[test]
    fn final_commits_only_the_remaining_suffix() {
        let mut tracker = StablePrefixTracker::default();
        let partial = commit_first_clause(&mut tracker);
        assert_eq!(partial.commit, "这是一个离线语音识别模型测试，");
        let final_result = tracker.update(&update(
            "这是一个离线语音识别模型测试，今天天气很好。",
            true,
        ));
        assert_eq!(final_result.commit, "今天天气很好。");
        assert!(final_result.unstable.is_empty());
    }

    #[test]
    fn does_not_treat_decimal_punctuation_as_a_commit_boundary() {
        let mut tracker = StablePrefixTracker::default();
        for _ in 0..STABILITY_HISTORY {
            let partial = tracker.update(&update("版本12.34完成后继续处理", false));
            assert!(partial.commit.is_empty());
        }
    }

    #[test]
    fn freezes_partial_commits_after_the_first_revision() {
        let mut tracker = StablePrefixTracker::default();
        let _ = commit_first_clause(&mut tracker);
        let revision = tracker.update(&update_with_revision(
            "这是一个离线语音识别模型测试，今天天气真好",
            true,
            1,
            false,
        ));
        assert!(!revision.conflict);
        assert!(revision.commit.is_empty());

        for _ in 0..STABILITY_HISTORY {
            let partial = tracker.update(&update_with_revision(
                "这是一个离线语音识别模型测试，今天天气真好，我们正在继续",
                false,
                1,
                false,
            ));
            assert!(partial.commit.is_empty());
        }

        let final_result = tracker.update(&update_with_revision(
            "这是一个离线语音识别模型测试，今天天气真好，我们正在继续。",
            false,
            1,
            true,
        ));
        assert_eq!(final_result.commit, "今天天气真好，我们正在继续。");
    }

    #[test]
    fn reports_conflict_when_revision_crosses_the_committed_prefix() {
        let mut tracker = StablePrefixTracker::default();
        let committed = commit_first_clause(&mut tracker);
        assert!(!committed.commit.is_empty());

        let revision = tracker.update(&update_with_revision(
            "那是一个离线语音识别模型测试，今天天气很好",
            true,
            1,
            false,
        ));
        assert!(revision.conflict);
        assert!(revision.commit.is_empty());
    }
}
