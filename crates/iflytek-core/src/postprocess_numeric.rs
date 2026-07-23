use std::collections::{HashMap, HashSet};

use crate::postprocess_types::{RenderedWord, WordKind};

#[derive(Clone, Debug)]
pub(crate) struct NumericPiece {
    source: String,
    text: String,
    category: Option<char>,
    kind: WordKind,
}

impl NumericPiece {
    fn new(
        source: impl Into<String>,
        text: impl Into<String>,
        category: Option<char>,
        kind: WordKind,
    ) -> Self {
        Self {
            source: source.into(),
            text: text.into(),
            category,
            kind,
        }
    }
}

pub(crate) fn numeric_features(tokens: &[String]) -> Vec<f32> {
    let mut features = Vec::new();
    for token in tokens {
        let multiple = token.chars().count() > 1;
        for (index, character) in token.chars().enumerate() {
            features.push(f32::from(multiple && is_numeric_feature_character(character)));
            features.push(f32::from(index > 0));
        }
    }
    features
}

pub(crate) fn convert_numeric(characters: &[char], labels: &[String]) -> Vec<NumericPiece> {
    let spans = find_spans(characters, labels);
    let mut pieces = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let Some(&(end, category)) = spans.get(&index) else {
            let source = characters[index].to_string();
            pieces.push(NumericPiece::new(
                source.clone(),
                source,
                None,
                WordKind::Text,
            ));
            index += 1;
            continue;
        };
        let source = characters[index..end].iter().collect::<String>();
        let text = match category {
            'A' => cardinal(&source),
            'B' if source.chars().all(is_sequence_digit) => digit_sequence(&source),
            'B' => {
                pieces.extend(source.chars().map(|character| {
                    let text = character.to_string();
                    NumericPiece::new(&text, text.clone(), None, WordKind::Text)
                }));
                index = end;
                continue;
            }
            'C' => source.clone(),
            'D' => time(&source),
            _ => source.clone(),
        };
        pieces.push(NumericPiece::new(
            source,
            text,
            Some(category),
            WordKind::Number,
        ));
        index = end;
    }
    pieces
}

pub(crate) fn apply_semantics(pieces: &[NumericPiece]) -> Vec<NumericPiece> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < pieces.len() {
        let mut piece = pieces[index].clone();
        if piece.category == Some('C') {
            if let Some(prefix) = prefix_symbol(&piece.source)
                && index + 1 < pieces.len()
                && matches!(pieces[index + 1].category, Some('A' | 'B'))
            {
                    let mut expression = vec![pieces[index + 1].clone()];
                    let mut end = index + 2;
                    while end < pieces.len() && pieces[end].category.is_some() {
                        expression.push(pieces[end].clone());
                        end += 1;
                    }
                    let mut rendered = String::new();
                    for expression_index in 0..expression.len() {
                        let item = &expression[expression_index];
                        if item.source == "分之" {
                            rendered.push('/');
                        } else if let Some(symbol) = symbol(&item.source) {
                            rendered.push_str(symbol);
                        } else if expression_index > 0
                            && expression[expression_index - 1].source == "点"
                        {
                            rendered.push_str(&digit_sequence(&item.source));
                        } else {
                            rendered.push_str(&item.text);
                        }
                    }
                    rendered.push_str(prefix);
                    let source = format!(
                        "{}{}",
                        piece.source,
                        expression.iter().map(|item| item.source.as_str()).collect::<String>()
                    );
                    output.push(NumericPiece::new(
                        source,
                        rendered,
                        expression.last().and_then(|item| item.category),
                        WordKind::Number,
                    ));
                    index = end;
                    continue;
            }
            if piece.source == "分之" && !output.is_empty() && index + 1 < pieces.len() {
                let previous = output.last().cloned().expect("output is not empty");
                let following = &pieces[index + 1];
                if matches!(previous.category, Some('A' | 'B'))
                    && matches!(following.category, Some('A' | 'B'))
                {
                    output.pop();
                    if previous.source == "万" {
                        output.push(NumericPiece::new(
                            format!("{}{}{}", previous.source, piece.source, following.source),
                            format!("万min之{}", following.text),
                            following.category,
                            WordKind::Number,
                        ));
                    } else {
                        output.push(NumericPiece::new(
                            previous.source,
                            following.text.clone(),
                            following.category,
                            WordKind::Number,
                        ));
                        output.push(NumericPiece::new(
                            piece.source,
                            "/",
                            Some('C'),
                            WordKind::Number,
                        ));
                        output.push(NumericPiece::new(
                            following.source.clone(),
                            previous.text,
                            previous.category,
                            WordKind::Number,
                        ));
                    }
                    index += 2;
                    continue;
                }
            }
            if let Some(symbol) = symbol(&piece.source) {
                if output.last().is_some_and(|previous| previous.text.ends_with('不')) {
                    output.push(piece);
                } else if piece.source == "点"
                    && !(output.last().is_some_and(|previous| {
                        matches!(previous.category, Some('A' | 'B'))
                    }) && index + 1 < pieces.len()
                        && matches!(pieces[index + 1].category, Some('A' | 'B')))
                {
                    piece.category = None;
                    piece.kind = WordKind::Text;
                    output.push(piece);
                } else {
                    piece.text = symbol.to_string();
                    piece.kind = WordKind::Number;
                    output.push(piece);
                }
                index += 1;
                continue;
            }
        }
        if output.last().is_some_and(|previous| previous.text == ".")
            && matches!(piece.category, Some('A' | 'B'))
        {
            piece.text = digit_sequence(&piece.source);
        }
        if output.last().is_some_and(|previous| matches!(previous.category, Some('A' | 'B')))
            && matches!(piece.category, Some('A' | 'B'))
        {
            let previous = output.last().expect("output is not empty");
            let adjacent_source = format!("{}{}", previous.source, piece.source);
            let mut run_start = output.len() - 1;
            while run_start > 0
                && matches!(output[run_start - 1].category, Some('A' | 'B'))
            {
                run_start -= 1;
            }
            let decimal_zero_prefix = previous.category == Some('A')
                && piece.category == Some('B')
                && run_start > 0
                && output[run_start - 1].text == "."
                && output[run_start..].iter().all(|item| {
                    item.source
                        .chars()
                        .all(|character| digit_value(character) == Some(0))
                });
            let mut run_end = index + 1;
            while run_end < pieces.len() && matches!(pieces[run_end].category, Some('A' | 'B')) {
                run_end += 1;
            }
            let temperature_singleton_run = run_start > 0
                && output[run_start - 1].text == "."
                && run_end < pieces.len()
                && matches!(pieces[run_end].source.as_str(), "摄氏度" | "华氏度")
                && output[run_start..]
                    .iter()
                    .chain(pieces[index..run_end].iter())
                    .all(|item| item.source.chars().count() == 1);
            let temperature_cardinal_suffix = previous.category == Some('B')
                && piece.category == Some('A')
                && index + 1 < pieces.len()
                && matches!(pieces[index + 1].source.as_str(), "摄氏度" | "华氏度");
            let joins_digit_sequence = adjacent_source.chars().all(is_sequence_digit)
                && (temperature_singleton_run
                    || !(temperature_cardinal_suffix
                        || (previous.category == Some('A')
                            && piece.category == Some('B')
                            && !decimal_zero_prefix)));
            if !joins_digit_sequence {
                piece.text.insert(0, ' ');
            }
        } else if output.last().is_some_and(|previous| {
            previous.category == Some('D')
                && (previous.text != previous.source
                    || previous.source.chars().all(is_numeric_feature_character)
                    || previous.source.chars().next_back().is_some_and(|character| {
                        digit_value(character).is_some()
                    }))
        }) && matches!(piece.category, Some('A' | 'B'))
        {
            piece.text.insert(0, ' ');
        }
        output.push(piece);
        index += 1;
    }
    output
}

pub(crate) fn merge_text_pieces(
    pieces: &[NumericPiece],
    token_boundaries: &HashSet<usize>,
) -> Vec<RenderedWord> {
    let mut expanded = Vec::new();
    let mut source_cursor = 0;
    for piece in pieces {
        let source_len = piece.source.chars().count();
        let source_end = source_cursor + source_len;
        let mut split_offsets = token_boundaries
            .iter()
            .filter(|boundary| source_cursor <= **boundary && **boundary + 1 < source_end)
            .map(|boundary| boundary - source_cursor + 1)
            .collect::<Vec<_>>();
        split_offsets.sort_unstable();
        if piece.text == piece.source && !split_offsets.is_empty() {
            let characters = piece.source.chars().collect::<Vec<_>>();
            let mut start = 0;
            for end in split_offsets.into_iter().chain(std::iter::once(source_len)) {
                let fragment = characters[start..end].iter().collect::<String>();
                expanded.push(NumericPiece::new(
                    &fragment,
                    fragment.clone(),
                    piece.category,
                    piece.kind,
                ));
                start = end;
            }
        } else {
            expanded.push(piece.clone());
        }
        source_cursor = source_end;
    }

    let mut words: Vec<RenderedWord> = Vec::new();
    let mut source_cursor = 0;
    for piece in expanded {
        let source_end = source_cursor + piece.source.chars().count();
        let kind = if piece.category.is_some() {
            WordKind::Number
        } else {
            piece.kind
        };
        if !words.is_empty()
            && source_cursor > 0
            && !token_boundaries.contains(&(source_cursor - 1))
        {
            let previous = words.last_mut().expect("words is not empty");
            previous.word.text.push_str(&piece.text);
            previous.word.kind = WordKind::Text;
            previous.source_end = source_end;
        } else {
            words.push(RenderedWord::new(piece.text, kind, source_end));
        }
        source_cursor = source_end;
    }
    words
}

pub(crate) fn coalesce_within_tokens(
    words: &[RenderedWord],
    token_boundaries: &HashSet<usize>,
) -> Vec<RenderedWord> {
    let mut output: Vec<RenderedWord> = Vec::new();
    for rendered in words {
        if output.last().is_some_and(|previous| {
            previous.source_end > 0 && !token_boundaries.contains(&(previous.source_end - 1))
        }) {
            let previous = output.last_mut().expect("output is not empty");
            previous.word.text.push_str(&rendered.word.text);
            previous.source_end = rendered.source_end;
        } else {
            output.push(rendered.clone());
        }
    }
    output
}

pub(crate) fn coalesce_per_mille(words: &[RenderedWord]) -> Vec<RenderedWord> {
    let mut output: Vec<RenderedWord> = Vec::new();
    for rendered in words {
        if rendered.word.text == "‰"
            && output
                .last()
                .is_some_and(|previous| is_unsigned_decimal(&previous.word.text))
        {
            let previous = output.last_mut().expect("output is not empty");
            previous.word.text.push('‰');
            previous.source_end = rendered.source_end;
        } else {
            output.push(rendered.clone());
        }
    }
    output
}

pub(crate) fn is_numeric_feature_character(character: char) -> bool {
    digit_value(character).is_some()
        || matches!(
            character,
            '十' | '拾' | '百' | '佰' | '千' | '仟' | '万' | '萬' | '亿' | '億' | '兆'
        )
}

pub(crate) fn is_sequence_digit(character: char) -> bool {
    character != '两' && digit_value(character).is_some()
}

pub(crate) fn is_digit_character(character: char) -> bool {
    digit_value(character).is_some()
}

pub(crate) fn numeric_source_character(character: char) -> bool {
    is_digit_character(character)
        || matches!(
            character,
            '十' | '百' | '千' | '万' | '亿' | '兆' | '点' | '时' | '分' | '秒'
                | '年' | '月' | '日' | '号' | '第' | '负' | '正' | '半' | '整' | '刻'
                | '加' | '减' | '等' | '于' | '大' | '小' | '比' | '之'
        )
}

pub(crate) fn has_prefix_symbol(text: &str) -> bool {
    ["百分之", "千分之", "万分之"]
        .iter()
        .any(|prefix| text.contains(prefix))
}

fn find_spans(characters: &[char], labels: &[String]) -> HashMap<usize, (usize, char)> {
    let mut spans = HashMap::new();
    let mut index = 0;
    while index < labels.len() {
        let label = labels[index].as_str();
        if let Some((category, middle, end_label)) = grouped_label(label) {
            if label == "BC"
                && index + 1 < labels.len()
                && labels[index + 1] == "O"
                && characters.get(index) == Some(&'分')
                && characters.get(index + 1) == Some(&'之')
            {
                spans.insert(index, (index + 2, category));
                index += 2;
                continue;
            }
            let mut end = index + 1;
            while end < labels.len() && matches!(labels[end].as_str(), value if value == middle || value == end_label) {
                let terminal = labels[end] == end_label;
                end += 1;
                if terminal {
                    break;
                }
            }
            spans.insert(index, (end, category));
            index = end;
            continue;
        }
        if let Some(category) = single_label(label) {
            spans.insert(index, (index + 1, category));
            index += 1;
            continue;
        }
        if label == "E" && characters.get(index) == Some(&'万') {
            spans.insert(index, (index + 1, 'A'));
        }
        index += 1;
    }
    spans
}

fn single_label(label: &str) -> Option<char> {
    match label {
        "SA" | "EA" => Some('A'),
        "SB" | "EB" => Some('B'),
        "SC" | "EC" => Some('C'),
        "SD" | "ED" => Some('D'),
        _ => None,
    }
}

fn grouped_label(label: &str) -> Option<(char, &'static str, &'static str)> {
    match label {
        "BA" | "MA" => Some(('A', "MA", "EA")),
        "BB" | "MB" => Some(('B', "MB", "EB")),
        "BC" | "MC" => Some(('C', "MC", "EC")),
        "BD" | "MD" => Some(('D', "MD", "ED")),
        _ => None,
    }
}

fn cardinal(source: &str) -> String {
    if let Some(suffix) = source.chars().next_back().filter(|suffix| {
        matches!(suffix, '万' | '萬' | '亿' | '億')
            && source.chars().filter(|character| character == suffix).count() == 1
    }) {
        let coefficient = &source[..source.len() - suffix.len_utf8()];
        if let Some(value) = parse_chinese_number(if coefficient.is_empty() { "一" } else { coefficient }) {
            let rendered = if value.abs() >= 1_000 {
                format_grouped(value)
            } else {
                value.to_string()
            };
            return format!("{}{}", rendered, suffix);
        }
    }
    parse_chinese_number(source)
        .map(format_number)
        .unwrap_or_else(|| source.to_string())
}

fn parse_chinese_number(source: &str) -> Option<i64> {
    if source.is_empty() {
        return None;
    }
    let has_unit = source.chars().any(|character| {
        matches!(
            character,
            '十' | '拾' | '百' | '佰' | '千' | '仟' | '万' | '萬' | '亿' | '億' | '兆'
        )
    });
    if !has_unit {
        let digits = source
            .chars()
            .map(digit_value)
            .collect::<Option<Vec<_>>>()?;
        return digits
            .into_iter()
            .try_fold(0_i64, |value, digit| value.checked_mul(10)?.checked_add(digit));
    }
    let mut total = 0_i64;
    let mut section = 0_i64;
    let mut number = None;
    for character in source.chars() {
        if let Some(digit) = digit_value(character) {
            number = Some(digit);
            continue;
        }
        if let Some(unit) = small_unit(character) {
            section = section.checked_add(number.unwrap_or(1).checked_mul(unit)?)?;
            number = None;
            continue;
        }
        if let Some(unit) = large_unit(character) {
            section = section.checked_add(number.unwrap_or(0))?;
            total = total.checked_add(section.checked_mul(unit)?)?;
            section = 0;
            number = None;
            continue;
        }
        return None;
    }
    total.checked_add(section)?.checked_add(number.unwrap_or(0))
}

fn format_number(value: i64) -> String {
    if value.abs() < 10_000 {
        return value.to_string();
    }
    format_grouped(value)
}

fn format_grouped(value: i64) -> String {
    let negative = value.is_negative();
    let digits = value.unsigned_abs().to_string();
    let mut output = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    if negative {
        output.insert(0, '-');
    }
    output
}

fn digit_sequence(source: &str) -> String {
    source
        .chars()
        .map(|character| {
            digit_value(character)
                .and_then(|value| char::from_digit(value as u32, 10))
                .unwrap_or(character)
        })
        .collect()
}

fn time(source: &str) -> String {
    let characters = source.chars().collect::<Vec<_>>();
    let Some(separator) = characters.iter().position(|character| matches!(character, '点' | '时')) else {
        return source.to_string();
    };
    if separator == 0 {
        return source.to_string();
    }
    let hour_source = characters[..separator].iter().collect::<String>();
    let hour = cardinal(&hour_source);
    if !hour.chars().all(|character| character.is_ascii_digit() || character == ',') {
        return source.to_string();
    }
    let mut minute_source = characters[separator + 1..].iter().collect::<String>();
    if minute_source.is_empty() || minute_source == "整" {
        return format!("{}{}", hour, characters[separator]);
    }
    let minute_has_unit = minute_source.ends_with('钟') || minute_source.ends_with('分');
    minute_source = minute_source
        .strip_suffix('钟')
        .or_else(|| minute_source.strip_suffix('分'))
        .unwrap_or(&minute_source)
        .to_string();
    if matches!(minute_source.as_str(), "零" | "〇" | "洞") && !minute_has_unit {
        return source.to_string();
    }
    let minute = match minute_source.as_str() {
        "半" => Some(30),
        "一刻" => Some(15),
        "三刻" => Some(45),
        _ => parse_chinese_number(&minute_source),
    };
    minute
        .map(|minute| format!("{}:{:02}", hour, minute))
        .unwrap_or_else(|| source.to_string())
}

fn digit_value(character: char) -> Option<i64> {
    match character {
        '零' | '〇' | '洞' => Some(0),
        '一' | '壹' | '幺' => Some(1),
        '二' | '贰' | '两' => Some(2),
        '三' | '叁' => Some(3),
        '四' | '肆' => Some(4),
        '五' | '伍' => Some(5),
        '六' | '陆' => Some(6),
        '七' | '柒' => Some(7),
        '八' | '捌' => Some(8),
        '九' | '玖' => Some(9),
        value if value.is_ascii_digit() => value.to_digit(10).map(i64::from),
        _ => None,
    }
}

fn small_unit(character: char) -> Option<i64> {
    match character {
        '十' | '拾' => Some(10),
        '百' | '佰' => Some(100),
        '千' | '仟' => Some(1_000),
        _ => None,
    }
}

fn large_unit(character: char) -> Option<i64> {
    match character {
        '万' | '萬' => Some(10_000),
        '亿' | '億' => Some(100_000_000),
        '兆' => Some(1_000_000_000_000),
        _ => None,
    }
}

fn symbol(source: &str) -> Option<&'static str> {
    match source {
        "点" => Some("."),
        "加" | "加上" => Some("+"),
        "减" | "减去" | "负" => Some("-"),
        "乘" | "乘以" => Some("×"),
        "除" | "除以" => Some("÷"),
        "等于" => Some("="),
        "大于" => Some(">"),
        "小于" => Some("<"),
        "比" => Some(":"),
        "摄氏度" => Some("℃"),
        "华氏度" => Some("℉"),
        _ => None,
    }
}

fn prefix_symbol(source: &str) -> Option<&'static str> {
    match source {
        "百分之" => Some("%"),
        "千分之" => Some("‰"),
        "万分之" => Some("/10000"),
        _ => None,
    }
}

fn is_unsigned_decimal(value: &str) -> bool {
    let mut point = false;
    let mut digit = false;
    for character in value.chars() {
        if character == '.' && !point {
            point = true;
        } else if character.is_ascii_digit() {
            digit = true;
        } else {
            return false;
        }
    }
    digit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chinese_cardinals_cover_large_and_sequence_forms() {
        assert_eq!(cardinal("一百二十三"), "123");
        assert_eq!(cardinal("一万零三十"), "10,030");
        assert_eq!(cardinal("一亿零三万零五"), "100,030,005");
        assert_eq!(cardinal("二零二六"), "2026");
        assert_eq!(cardinal("一千万"), "1,000万");
    }

    #[test]
    fn time_conversion_handles_quarters_and_midnight() {
        assert_eq!(time("三点一刻"), "3:15");
        assert_eq!(time("零点零五分"), "0:05");
    }
}
