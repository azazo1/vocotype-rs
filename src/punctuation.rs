enum ConvertedChar {
    Converted(&'static str),
    Raw(char),
}

pub fn convert_to_english_punctuation(input: &str) -> Option<String> {
    let mut changed = false;
    let mut intermediate = String::with_capacity(input.len());

    for ch in input.chars() {
        match convert_char(ch) {
            ConvertedChar::Converted(value) => {
                changed = true;
                intermediate.push_str(value);
            }
            ConvertedChar::Raw(value) => intermediate.push(value),
        }
    }

    changed.then(|| merge_spacing_markers(&intermediate))
}

pub fn strip_trailing_period(text: &str) -> String {
    let trimmed = text.trim_end();
    let Some(last) = trimmed.chars().next_back() else {
        return text.to_string();
    };
    if !is_period(last) {
        return text.to_string();
    }
    trimmed[..trimmed.len() - last.len_utf8()]
        .trim_end()
        .to_string()
}

fn convert_char(ch: char) -> ConvertedChar {
    use ConvertedChar::*;

    match ch {
        '》' => Converted("\0\0>\0"),
        '《' => Converted("\0<\0\0"),
        '：' => Converted("\0\0:\0"),
        '；' => Converted("\0\0;\0"),
        '“' => Converted("\0\"\0\0"),
        '”' => Converted("\0\0\"\0"),
        '‘' => Converted("\0'\0\0"),
        '’' => Converted("\0\0'\0"),
        '！' => Converted("\0\0!\0"),
        '…' => Converted("\0\0...\0\0"),
        '（' => Converted("\0(\0\0"),
        '）' => Converted("\0\0)\0"),
        '【' => Converted("\0[\0\0"),
        '】' => Converted("\0\0]\0"),
        '、' => Converted("\0\0,\0"),
        '。' => Converted("\0\0.\0"),
        '，' => Converted("\0\0,\0"),
        '？' => Converted("\0\0?\0"),
        _ => Raw(ch),
    }
}

fn merge_spacing_markers(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    let mut prev_is_whitespace = false;

    while cursor < chars.len() {
        let mut marker_count = 0;
        while cursor < chars.len() && chars[cursor] == '\0' {
            marker_count += 1;
            cursor += 1;
        }

        if marker_count == 0 {
            output.push(chars[cursor]);
            prev_is_whitespace = chars[cursor].is_whitespace();
            cursor += 1;
        } else if marker_count == 1
            && !prev_is_whitespace
            && cursor < chars.len()
            && !chars[cursor].is_whitespace()
        {
            output.push(' ');
            prev_is_whitespace = true;
        }
    }

    output
}

fn is_period(ch: char) -> bool {
    matches!(ch, '.' | '。' | '．')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_common_chinese_punctuation() {
        let input = concat!(
            "你好，",
            "世界！",
            "“Rust”",
            "（2026）",
            "。"
        );
        let output = convert_to_english_punctuation(input).unwrap();

        assert_eq!(output, "你好, 世界!\"Rust\"(2026).");
    }

    #[test]
    fn converts_adjacent_quotes_with_reference_spacing() {
        let input = concat!(
            "我喜欢",
            "‘CLI’",
            "和",
            "“Rust”"
        );
        let output = convert_to_english_punctuation(input).unwrap();

        assert_eq!(output, "我喜欢 'CLI' 和 \"Rust\"");
    }

    #[test]
    fn returns_none_when_no_punctuation_changed() {
        assert!(convert_to_english_punctuation("hello, world.").is_none());
    }

    #[test]
    fn strips_only_trailing_period() {
        assert_eq!(strip_trailing_period("你好。"), "你好");
        assert_eq!(strip_trailing_period("hello.  "), "hello");
        assert_eq!(strip_trailing_period("你好！"), "你好！");
    }
}
