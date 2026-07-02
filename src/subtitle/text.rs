const DEFAULT_MAX_CHARS: usize = 24;

#[derive(Clone, Debug)]
pub struct TimedTextGroup {
    pub text: String,
    pub start_seconds: f32,
    pub end_seconds: f32,
}

pub fn split_text_for_subtitle(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(8);
    let mut output = Vec::new();
    for sentence in split_sentences(text) {
        split_long_sentence(&sentence, max_chars, &mut output);
    }
    output
}

pub fn split_timed_text_parts(
    text: &str,
    raw_text: &str,
    tokens: &[String],
    timestamps: &[f32],
    max_chars: usize,
    segment_end_seconds: f32,
) -> Vec<TimedTextGroup> {
    let text_parts = split_text_for_subtitle(text, max_chars);
    let raw_total = compact_text(raw_text).chars().count();
    if text_parts.is_empty() || raw_total == 0 {
        return Vec::new();
    }

    let timed_tokens = tokens
        .iter()
        .zip(timestamps.iter().copied())
        .filter_map(|(token, timestamp)| {
            let text = compact_text(token);
            (!text.is_empty()).then_some((text, timestamp))
        })
        .collect::<Vec<_>>();
    if timed_tokens.is_empty() {
        return Vec::new();
    }

    let token_step = estimate_token_step(&timed_tokens);
    let mut output = Vec::new();
    let mut token_index = 0_usize;
    for part in text_parts {
        let target_len = compact_text(&part).chars().count();
        if target_len == 0 || token_index >= timed_tokens.len() {
            continue;
        }

        let start_index = token_index;
        let mut collected = 0_usize;
        while token_index < timed_tokens.len() && collected < target_len {
            collected += timed_tokens[token_index].0.chars().count();
            token_index += 1;
        }
        let end_index = token_index.saturating_sub(1);
        let end_seconds = timed_tokens
            .get(token_index)
            .map(|(_, timestamp)| *timestamp)
            .unwrap_or_else(|| {
                (timed_tokens[end_index].1 + token_step).min(segment_end_seconds)
            });
        output.push(TimedTextGroup {
            text: part,
            start_seconds: timed_tokens[start_index].1,
            end_seconds: end_seconds.max(timed_tokens[start_index].1),
        });
    }
    output
}

fn estimate_token_step(tokens: &[(String, f32)]) -> f32 {
    let mut intervals = tokens
        .windows(2)
        .filter_map(|window| {
            let delta = window[1].1 - window[0].1;
            (delta > 0.0).then_some(delta)
        })
        .collect::<Vec<_>>();
    if intervals.is_empty() {
        return 0.3;
    }
    intervals.sort_by(|left, right| left.total_cmp(right));
    intervals[intervals.len() / 2].clamp(0.08, 0.8)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut current = String::new();
    for ch in text.trim().chars() {
        current.push(ch);
        if is_sentence_punctuation(ch) {
            push_trimmed(&mut output, &mut current);
        }
    }
    push_trimmed(&mut output, &mut current);
    output
}

fn split_long_sentence(sentence: &str, max_chars: usize, output: &mut Vec<String>) {
    let mut current = String::new();
    for ch in sentence.chars() {
        current.push(ch);
        let should_split = current.chars().count() >= max_chars
            && (is_soft_break(ch) || current.chars().count() >= max_chars + 8);
        if should_split {
            push_trimmed(output, &mut current);
        }
    }
    push_trimmed(output, &mut current);
}

fn push_trimmed(output: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        output.push(trimmed.to_string());
    }
    current.clear();
}

fn compact_text(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && !is_sentence_punctuation(*ch))
        .collect()
}

fn is_sentence_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '.' | '?'
            | '!'
            | ','
            | ';'
            | ':'
            | '\u{ff0c}'
            | '\u{3002}'
            | '\u{ff1f}'
            | '\u{ff01}'
            | '\u{ff1b}'
            | '\u{ff1a}'
    )
}

fn is_soft_break(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, ',' | ';' | ':' | '\u{ff0c}' | '\u{ff1b}' | '\u{ff1a}')
}

pub fn default_max_chars() -> usize {
    DEFAULT_MAX_CHARS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_chinese_punctuated_text() {
        let parts = split_text_for_subtitle("你好, 我是测试。下一句来了。", 24);
        assert_eq!(parts, vec!["你好,", "我是测试。", "下一句来了。"]);
    }

    #[test]
    fn splits_long_text_without_punctuation() {
        let parts = split_text_for_subtitle("这是一个没有标点但是很长很长的字幕文本", 10);
        assert!(parts.len() > 1);
    }

    #[test]
    fn maps_punctuated_parts_to_token_timestamps() {
        let tokens = ["你", "好", "世", "界"]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let groups = split_timed_text_parts(
            "你好，世界。",
            "你好世界",
            &tokens,
            &[0.1, 0.3, 1.1, 1.3],
            24,
            2.0,
        );
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].text, "你好，");
        assert_eq!(groups[0].start_seconds, 0.1);
        assert_eq!(groups[0].end_seconds, 1.1);
        assert_eq!(groups[1].text, "世界。");
        assert_eq!(groups[1].start_seconds, 1.1);
    }
}
