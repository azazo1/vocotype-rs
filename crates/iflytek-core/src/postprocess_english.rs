use crate::postprocess_types::{RenderedWord, WordKind};

pub(crate) fn format_english_words(words: &[RenderedWord]) -> Vec<RenderedWord> {
    let mut output: Vec<RenderedWord> = Vec::new();
    for rendered in words {
        let mut word = rendered.word.clone();
        if is_ascii_letters(&word.text) {
            let previous = output.last().map_or("", |item| item.word.text.as_str());
            let previous_stripped = previous.trim();
            let previous_last = previous_stripped.chars().next_back();
            if output.is_empty() && word.text.starts_with(|character: char| character.is_ascii_lowercase()) {
                uppercase_first(&mut word.text);
            }
            let join_initials = word.text.chars().count() == 1
                && word.text.chars().all(|character| character.is_ascii_uppercase())
                && previous_stripped.chars().count() == 1
                && previous_stripped
                    .chars()
                    .all(|character| character.is_ascii_uppercase());
            let follows_chinese = previous_last.is_some_and(is_cjk);
            if output.is_empty() || (!follows_chinese && !join_initials) {
                word.text.insert(0, ' ');
            }
        } else if !word.text.trim().is_empty() {
            let stripped = word.text.trim_start().to_string();
            let mut leading = word.text[..word.text.len() - stripped.len()].to_string();
            if output.is_empty()
                && stripped
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_alphabetic())
            {
                let mut adjusted = stripped;
                uppercase_first(&mut adjusted);
                leading = " ".to_string();
                word.text = leading + &adjusted;
            } else {
                if !output.is_empty()
                    && stripped
                        .chars()
                        .next()
                        .is_some_and(|character| character.is_ascii_digit())
                    && output
                        .last()
                        .is_some_and(|previous| is_ascii_letters(previous.word.text.trim()))
                {
                    leading = " ".to_string();
                }
                word.text = leading + &stripped;
            }
        }
        output.push(RenderedWord {
            word,
            source_end: rendered.source_end,
        });
    }
    output
}

pub(crate) fn apply_english_grammar(
    raw_tokens: &[String],
    words: &[RenderedWord],
    punctuation_labels: &[usize],
) -> Vec<RenderedWord> {
    if raw_tokens.is_empty() || raw_tokens.iter().any(|token| !is_ascii_letters(token)) {
        return words.to_vec();
    }
    let source_ends = token_source_ends(raw_tokens);
    let mut output = Vec::new();
    let mut token_start = 0;
    let mut source_start = 0;
    for (token_end, source_end) in source_ends.iter().copied().enumerate() {
        let label = punctuation_labels
            .get(source_end.saturating_sub(1))
            .copied()
            .unwrap_or(0);
        if !(1..=4).contains(&label) && token_end + 1 != raw_tokens.len() {
            continue;
        }
        let relative_words = words
            .iter()
            .filter(|item| source_start < item.source_end && item.source_end <= source_end)
            .map(|item| RenderedWord {
                word: item.word.clone(),
                source_end: item.source_end - source_start,
            })
            .collect::<Vec<_>>();
        let rendered_span = apply_english_grammar_span(
            &raw_tokens[token_start..=token_end],
            &relative_words,
        );
        for (span_index, mut item) in rendered_span.into_iter().enumerate() {
            if !output.is_empty() && span_index == 0 && item.word.text.starts_with(' ') {
                item.word.text.remove(0);
            }
            item.source_end += source_start;
            output.push(item);
        }
        token_start = token_end + 1;
        source_start = source_end;
    }
    output
}

pub(crate) fn apply_spoken_web_semantics(
    raw_tokens: &[String],
    words: &[RenderedWord],
) -> Vec<RenderedWord> {
    let Some(first) = raw_tokens.first() else {
        return words.to_vec();
    };
    if !first.starts_with("网址") && !first.starts_with("邮箱") {
        return words.to_vec();
    }
    let source_ends = token_source_ends(raw_tokens);
    let last_point = raw_tokens.iter().rposition(|token| token == "点");
    let mut output = vec![RenderedWord::new(first, WordKind::Text, source_ends[0])];
    let mut index = 1;
    while index < raw_tokens.len() {
        let token = &raw_tokens[index];
        if token.chars().count() == 1 && token.chars().all(|character| character.is_ascii_uppercase()) {
            let mut end = index + 1;
            while end < raw_tokens.len()
                && raw_tokens[end].chars().count() == 1
                && raw_tokens[end]
                    .chars()
                    .all(|character| character.is_ascii_uppercase())
            {
                end += 1;
            }
            output.push(RenderedWord::new(
                raw_tokens[index..end].concat(),
                WordKind::Text,
                source_ends[end - 1],
            ));
            index = end;
            continue;
        }
        if Some(index) == last_point
            && index + 1 < raw_tokens.len()
            && is_ascii_letters(&raw_tokens[index + 1])
        {
            output.push(RenderedWord::new(
                format!(".{}", raw_tokens[index + 1]),
                WordKind::Text,
                source_ends[index + 1],
            ));
            index += 2;
            continue;
        }
        output.push(RenderedWord::new(
            token,
            WordKind::Text,
            source_ends[index],
        ));
        index += 1;
    }
    output
}

fn apply_english_grammar_span(
    raw_tokens: &[String],
    words: &[RenderedWord],
) -> Vec<RenderedWord> {
    if raw_tokens.is_empty() || raw_tokens.iter().any(|token| !is_ascii_letters(token)) {
        return words.to_vec();
    }
    let lowered = raw_tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let source_ends = token_source_ends(raw_tokens);
    let rendered = |text: String, source_end: usize, kind: WordKind| {
        RenderedWord::new(text, kind, source_end)
    };

    if lowered.first().is_some_and(|token| token == "call")
        && lowered[1..].iter().all(|token| english_digit(token).is_some())
    {
        let digits = lowered[1..]
            .iter()
            .filter_map(|token| english_digit(token))
            .collect::<String>();
        return vec![
            rendered(" Call".to_string(), source_ends[0], WordKind::Text),
            rendered(format!(" {}", digits), *source_ends.last().unwrap(), WordKind::Number),
        ];
    }

    if is_english_month(&lowered[0]) && lowered.len() >= 4 {
        for split in 2..lowered.len() {
            let ordinal = parse_english_ordinal(&lowered[1..split]);
            let year = parse_spoken_year(&lowered[split..]);
            if raw_tokens[0]
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
                && let (Some(ordinal), Some(year)) = (ordinal, year)
            {
                return vec![
                    rendered(
                        format!(" {}", capitalize(&lowered[0])),
                        source_ends[0],
                        WordKind::Text,
                    ),
                    rendered(
                        format!(" {}{}", ordinal, ordinal_suffix(ordinal)),
                        source_ends[split - 1],
                        WordKind::Number,
                    ),
                    rendered(
                        ", ".to_string(),
                        source_ends[split - 1],
                        WordKind::Punctuation,
                    ),
                    rendered(
                        year.to_string(),
                        *source_ends.last().unwrap(),
                        WordKind::Number,
                    ),
                ];
            }
            if let Some(ordinal) = ordinal {
                if split == 2 && is_ordinal_word(&lowered[1]) {
                    let month_day = rendered(
                        format!(" {} {}", capitalize(&lowered[0]), ordinal),
                        source_ends[1],
                        WordKind::Text,
                    );
                    if let Some(year) = year.filter(|year| (1000..=9999).contains(year)) {
                        return vec![
                            month_day,
                            rendered(
                                format!(" {}", year),
                                *source_ends.last().unwrap(),
                                WordKind::Number,
                            ),
                        ];
                    }
                    let mut output = vec![month_day];
                    output.extend((split..lowered.len()).map(|index| {
                        rendered(
                            format!(" {}", lowered[index]),
                            source_ends[index],
                            WordKind::Text,
                        )
                    }));
                    return output;
                }
                if year.is_some_and(|year| (1000..=9999).contains(&year)) {
                    let mut output = (0..split)
                        .map(|index| {
                            rendered(
                                format!(
                                    " {}",
                                    if index == 0 {
                                        capitalize(&lowered[index])
                                    } else {
                                        lowered[index].clone()
                                    }
                                ),
                                source_ends[index],
                                WordKind::Text,
                            )
                        })
                        .collect::<Vec<_>>();
                    output.push(rendered(
                        format!(" {}", year.unwrap()),
                        *source_ends.last().unwrap(),
                        WordKind::Number,
                    ));
                    return output;
                }
            }
        }
    }

    if lowered.len() >= 4
        && matches!(&lowered[lowered.len() - 2..], [left, right] if (left == "a" || left == "p") && right == "m")
    {
        let content_end = lowered.len() - 2;
        if lowered[..content_end] == ["nine", "eleven"] {
            return vec![
                rendered(" 911".to_string(), source_ends[content_end - 1], WordKind::Number),
                rendered(
                    format!(" {}{}", lowered[content_end], lowered[content_end + 1]),
                    *source_ends.last().unwrap(),
                    WordKind::Text,
                ),
            ];
        }
        for split in 1..content_end {
            let hour = parse_english_cardinal(&lowered[..split]);
            let minute = parse_english_cardinal(&lowered[split..content_end]);
            if hour.is_some_and(|hour| (0..=23).contains(&hour))
                && minute.is_some_and(|minute| (1..=59).contains(&minute))
            {
                return vec![
                    rendered(
                        format!("{}:{:02}", hour.unwrap(), minute.unwrap()),
                        source_ends[content_end - 1],
                        WordKind::Number,
                    ),
                    rendered(
                        format!(" {}{}", lowered[content_end], lowered[content_end + 1]),
                        *source_ends.last().unwrap(),
                        WordKind::Text,
                    ),
                ];
            }
            if hour.is_some_and(|hour| hour >= 10)
                && matches!(&lowered[split..content_end], [value] if value == "zero" || value == "oh")
            {
                return vec![
                    rendered(
                        hour.unwrap().to_string(),
                        source_ends[split - 1],
                        WordKind::Number,
                    ),
                    rendered(
                        format!(" {}", lowered[split]),
                        source_ends[content_end - 1],
                        WordKind::Text,
                    ),
                    rendered(
                        format!(" {}{}", lowered[content_end], lowered[content_end + 1]),
                        *source_ends.last().unwrap(),
                        WordKind::Text,
                    ),
                ];
            }
        }
        if lowered[..content_end]
            .iter()
            .all(|token| english_digit(token).is_some())
        {
            let digits = lowered[..content_end]
                .iter()
                .filter_map(|token| english_digit(token))
                .collect::<String>();
            return vec![
                rendered(digits, source_ends[content_end - 1], WordKind::Number),
                rendered(
                    format!(" {}{}", lowered[content_end], lowered[content_end + 1]),
                    *source_ends.last().unwrap(),
                    WordKind::Text,
                ),
            ];
        }
    }

    if lowered.last().is_some_and(|token| token == "percent")
        && let Some(value) = parse_english_cardinal(&lowered[..lowered.len() - 1])
    {
            if value > 100 {
                return vec![
                    rendered(
                        value.to_string(),
                        source_ends[source_ends.len() - 2],
                        WordKind::Number,
                    ),
                    rendered(
                        " percent".to_string(),
                        *source_ends.last().unwrap(),
                        WordKind::Text,
                    ),
                ];
            }
            return vec![
                rendered(
                    value.to_string(),
                    source_ends[source_ends.len() - 2],
                    WordKind::Number,
                ),
                rendered(
                    "%".to_string(),
                    *source_ends.last().unwrap(),
                    WordKind::Number,
                ),
            ];
    }

    if let Some(point) = lowered.iter().position(|token| token == "point") {
        let whole = parse_english_cardinal(&lowered[..point]);
        let decimal = &lowered[point + 1..];
        if let Some(whole) = whole
            && !decimal.is_empty()
            && decimal.iter().all(|token| english_digit(token).is_some())
        {
            return vec![
                rendered(
                    whole.to_string(),
                    source_ends[point - 1],
                    WordKind::Number,
                ),
                rendered(".".to_string(), source_ends[point], WordKind::Number),
                rendered(
                    decimal
                        .iter()
                        .filter_map(|token| english_digit(token))
                        .collect(),
                    *source_ends.last().unwrap(),
                    WordKind::Number,
                ),
            ];
        }
    }

    if lowered.len() >= 2 {
        if let Some(denominator) = fraction_denominator(lowered.last().unwrap())
            && let Some(numerator) = parse_english_cardinal(&lowered[..lowered.len() - 1])
        {
            return vec![rendered(
                format!("{}/{}", numerator, denominator),
                *source_ends.last().unwrap(),
                WordKind::Number,
            )];
        }
        if is_ordinal_word(lowered.last().unwrap()) {
            let mut ordinal_words = lowered.as_slice();
            let mut output = Vec::new();
            if lowered[0] == "the" && lowered.len() > 2 {
                output.push(rendered(
                    " The".to_string(),
                    source_ends[0],
                    WordKind::Text,
                ));
                ordinal_words = &lowered[1..];
            }
            if let Some(ordinal) = parse_english_ordinal(ordinal_words) {
                let leading = if output.is_empty() { "" } else { " " };
                output.push(rendered(
                    format!("{}{}{}", leading, ordinal, ordinal_suffix(ordinal)),
                    *source_ends.last().unwrap(),
                    WordKind::Number,
                ));
                return output;
            }
        }
    }

    if lowered.len() > 1 && lowered.iter().all(|token| english_digit(token).is_some()) {
        return vec![rendered(
            lowered
                .iter()
                .filter_map(|token| english_digit(token))
                .collect(),
            *source_ends.last().unwrap(),
            WordKind::Number,
        )];
    }
    if let Some(cardinal) = parse_english_cardinal(&lowered)
        && (lowered.len() > 1 || cardinal.abs() >= 10)
    {
        return vec![rendered(
            cardinal.to_string(),
            *source_ends.last().unwrap(),
            WordKind::Number,
        )];
    }
    if let Some(grouped) = parse_spoken_year(&lowered)
        && lowered.len() > 1
    {
        return vec![rendered(
            grouped.to_string(),
            *source_ends.last().unwrap(),
            WordKind::Number,
        )];
    }
    words.to_vec()
}

fn parse_english_cardinal(words: &[String]) -> Option<i64> {
    if words.is_empty() {
        return None;
    }
    let mut total = 0_i64;
    let mut current = 0_i64;
    let mut consumed = false;
    for word in words {
        if word == "and" {
            continue;
        }
        if let Some(value) = english_cardinal_value(word) {
            current += value;
            consumed = true;
        } else if word == "hundred" {
            current = current.max(1) * 100;
            consumed = true;
        } else if word == "thousand" {
            total += current.max(1) * 1_000;
            current = 0;
            consumed = true;
        } else if word == "million" {
            total += current.max(1) * 1_000_000;
            current = 0;
            consumed = true;
        } else {
            return None;
        }
    }
    consumed.then_some(total + current)
}

fn parse_spoken_year(words: &[String]) -> Option<i64> {
    parse_english_cardinal(words).or_else(|| {
        if words.len() < 2 {
            return None;
        }
        let century = parse_english_cardinal(&words[..1])?;
        let remainder = parse_english_cardinal(&words[1..])?;
        (1..=99)
            .contains(&century)
            .then_some(century * 100 + remainder)
    })
}

fn parse_english_ordinal(words: &[String]) -> Option<i64> {
    let (last, prefix) = words.split_last()?;
    let ordinal = ordinal_value(last)?;
    if prefix.is_empty() {
        return Some(ordinal);
    }
    if matches!(last.as_str(), "hundredth" | "thousandth" | "millionth") {
        let multiplier = ordinal;
        return Some(parse_english_cardinal(prefix)? * multiplier);
    }
    Some(parse_english_cardinal(prefix)? + ordinal)
}

fn english_cardinal_value(word: &str) -> Option<i64> {
    match word {
        "zero" | "oh" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => None,
    }
}

fn ordinal_value(word: &str) -> Option<i64> {
    match word {
        "first" => Some(1),
        "second" => Some(2),
        "third" => Some(3),
        "fourth" => Some(4),
        "fifth" => Some(5),
        "sixth" => Some(6),
        "seventh" => Some(7),
        "eighth" => Some(8),
        "ninth" => Some(9),
        "tenth" => Some(10),
        "eleventh" => Some(11),
        "twelfth" => Some(12),
        "thirteenth" => Some(13),
        "fourteenth" => Some(14),
        "fifteenth" => Some(15),
        "sixteenth" => Some(16),
        "seventeenth" => Some(17),
        "eighteenth" => Some(18),
        "nineteenth" => Some(19),
        "twentieth" => Some(20),
        "thirtieth" => Some(30),
        "fortieth" => Some(40),
        "fiftieth" => Some(50),
        "sixtieth" => Some(60),
        "seventieth" => Some(70),
        "eightieth" => Some(80),
        "ninetieth" => Some(90),
        "hundredth" => Some(100),
        "thousandth" => Some(1_000),
        "millionth" => Some(1_000_000),
        _ => None,
    }
}

fn english_digit(word: &str) -> Option<char> {
    match word {
        "zero" | "oh" => Some('0'),
        "one" => Some('1'),
        "two" => Some('2'),
        "three" => Some('3'),
        "four" => Some('4'),
        "five" => Some('5'),
        "six" => Some('6'),
        "seven" => Some('7'),
        "eight" => Some('8'),
        "nine" => Some('9'),
        _ => None,
    }
}

fn is_english_month(word: &str) -> bool {
    matches!(
        word,
        "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

fn is_ordinal_word(word: &str) -> bool {
    ordinal_value(word).is_some()
}

fn fraction_denominator(word: &str) -> Option<i64> {
    match word {
        "half" => Some(2),
        "third" => Some(3),
        "quarter" => Some(4),
        _ => None,
    }
}

fn ordinal_suffix(value: i64) -> &'static str {
    if (10..=20).contains(&(value % 100)) {
        "th"
    } else {
        match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    }
}

fn token_source_ends(tokens: &[String]) -> Vec<usize> {
    let mut cursor = 0;
    tokens
        .iter()
        .map(|token| {
            cursor += token.chars().count();
            cursor
        })
        .collect()
}

fn is_ascii_letters(value: &str) -> bool {
    !value.is_empty() && value.chars().all(|character| character.is_ascii_alphabetic())
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x9fff)
}

fn uppercase_first(value: &mut String) {
    if let Some(first) = value.chars().next() {
        let replacement = first.to_ascii_uppercase().to_string();
        value.replace_range(..first.len_utf8(), &replacement);
    }
}

fn capitalize(value: &str) -> String {
    let mut output = value.to_string();
    uppercase_first(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_cardinal_and_ordinal_parser_cover_compounds() {
        let words = ["one", "hundred", "and", "twenty", "three"]
            .map(str::to_string);
        assert_eq!(parse_english_cardinal(&words), Some(123));
        let ordinal = ["twenty", "second"].map(str::to_string);
        assert_eq!(parse_english_ordinal(&ordinal), Some(22));
    }

    #[test]
    fn english_spacing_joins_initials() {
        let words = vec![
            RenderedWord::new("G", WordKind::Text, 1),
            RenderedWord::new("P", WordKind::Text, 2),
            RenderedWord::new("T", WordKind::Text, 3),
        ];
        let formatted = format_english_words(&words);
        assert_eq!(
            formatted
                .iter()
                .map(|item| item.word.text.as_str())
                .collect::<String>(),
            " GPT"
        );
    }
}
