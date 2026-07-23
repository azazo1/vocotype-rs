use std::collections::HashSet;
use std::path::Path;

use anyhow::{Result, bail};

use crate::postprocess_english::{
    apply_english_grammar, apply_spoken_web_semantics, format_english_words,
};
use crate::postprocess_models::{NumericModel, PunctuationModel};
use crate::postprocess_numeric::{
    apply_semantics, coalesce_per_mille, coalesce_within_tokens, convert_numeric,
    has_prefix_symbol, is_digit_character, is_numeric_feature_character, is_sequence_digit,
    merge_text_pieces, numeric_features, numeric_source_character,
};
use crate::postprocess_tables::{NumericNotChangeList, ReplacementTable};
use crate::postprocess_types::{ProcessedWord, RenderedWord, WordKind};

#[derive(Clone, Debug, Default)]
pub struct PostprocessOptions {
    pub english_punctuation: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PostprocessResult {
    pub text: String,
    pub tokens: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Postprocessor {
    options: PostprocessOptions,
}

impl Postprocessor {
    pub fn new(options: PostprocessOptions) -> Self {
        Self { options }
    }

    pub fn apply(&self, text: &str, tokens: Vec<String>) -> PostprocessResult {
        let text = apply_output_options(text.trim().to_string(), &self.options);
        PostprocessResult { text, tokens }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeEsrPostprocessor {
    numeric: NumericModel,
    punctuation: PunctuationModel,
    replacements: ReplacementTable,
    not_change: NumericNotChangeList,
    options: PostprocessOptions,
}

impl EdgeEsrPostprocessor {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        number_normalization: &Path,
        number_vocabulary: &Path,
        number_not_change: &Path,
        punctuation_bert: &Path,
        punctuation_vocabulary: &Path,
        punctuation_maplist: &Path,
        replacements: &Path,
        options: PostprocessOptions,
    ) -> Result<Self> {
        Ok(Self {
            numeric: NumericModel::load(number_normalization, number_vocabulary)?,
            punctuation: PunctuationModel::load(
                punctuation_bert,
                punctuation_vocabulary,
                punctuation_maplist,
            )?,
            replacements: ReplacementTable::load(replacements)?,
            not_change: NumericNotChangeList::load(number_not_change)?,
            options,
        })
    }

    pub fn process(&self, tokens: &[String], final_input: bool) -> Result<PostprocessResult> {
        let raw_tokens = tokens
            .iter()
            .filter(|token| !token.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        let raw_text = raw_tokens.concat();
        let characters = raw_tokens
            .iter()
            .flat_map(|token| token.chars())
            .collect::<Vec<_>>();
        if characters.is_empty() {
            return Ok(PostprocessResult::default());
        }
        let token_boundaries = token_boundaries(&raw_tokens);
        let mut numeric_labels = self.numeric_labels(&raw_tokens, &characters)?;
        self.not_change.protect(&raw_text, &mut numeric_labels);
        let mut punctuation_labels = self.punctuation_labels(&raw_tokens, final_input)?;
        let semantic_labels = characters
            .iter()
            .zip(&numeric_labels)
            .map(|(character, label)| {
                if character.is_ascii() {
                    "O".to_string()
                } else {
                    label.clone()
                }
            })
            .collect::<Vec<_>>();
        let pieces = apply_semantics(&convert_numeric(&characters, &semantic_labels));
        let mut words = merge_text_pieces(&pieces, &token_boundaries);
        words = self.replacements.apply_words(&words)?;
        words = coalesce_within_tokens(&words, &token_boundaries);
        if raw_text.contains("零点") {
            for rendered in &mut words {
                rendered.word.text = normalize_midnight_zero(&rendered.word.text);
            }
        }
        words = format_english_words(&words);
        if raw_text
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic())
            && words
                .first()
                .is_some_and(|word| !word.word.text.starts_with(' '))
        {
            words[0].word.text.insert(0, ' ');
        }
        words = apply_english_grammar(&raw_tokens, &words, &punctuation_labels);
        words = apply_spoken_web_semantics(&raw_tokens, &words);
        words = coalesce_per_mille(&words);

        let rendered_text = joined_text(&words);
        let chinese_text_count = rendered_text.chars().filter(|character| is_cjk(*character)).count();
        let final_source_has_text = raw_tokens.last().is_some_and(|token| {
            token
                .chars()
                .any(|character| is_cjk(character) && !numeric_source_character(character))
        });
        let chinese_punctuation = chinese_text_count >= 1
            || rendered_text.chars().next_back().is_some_and(is_cjk)
            || final_source_has_text
            || ["‰", "÷", "℃", "℉"]
                .iter()
                .any(|symbol| rendered_text.contains(symbol));
        if final_input
            && !punctuation_labels.is_empty()
            && raw_text.chars().any(|character| character.is_ascii_alphabetic())
            && !raw_text.chars().any(is_cjk)
            && let Some(last) = punctuation_labels.last_mut()
        {
            *last = 2;
        }
        if should_suppress_final_punctuation(
            &raw_text,
            &rendered_text,
            &numeric_labels,
            &raw_tokens,
        ) && let Some(last) = punctuation_labels.last_mut()
        {
            *last = 0;
        }
        let mut words = punctuate(
            &words,
            &punctuation_labels,
            &token_boundaries,
            chinese_punctuation,
            final_input,
            characters.len(),
        );
        apply_word_options(&mut words, &self.options);
        let text = apply_output_options(joined_processed_text(&words), &self.options);
        Ok(PostprocessResult {
            text,
            tokens: words.into_iter().map(|word| word.text).collect(),
        })
    }

    fn numeric_labels(&self, raw_tokens: &[String], characters: &[char]) -> Result<Vec<String>> {
        let features = numeric_features(raw_tokens);
        if features.len() != characters.len() * 2 {
            bail!("numeric feature count differs from source characters")
        }
        let mut labels = Vec::with_capacity(characters.len());
        for offset in (0..characters.len()).step_by(NumericModel::MAX_LENGTH) {
            let end = (offset + NumericModel::MAX_LENGTH).min(characters.len());
            let tokens = characters[offset..end]
                .iter()
                .map(char::to_string)
                .collect::<Vec<_>>();
            labels.extend(self.numeric.predict(
                &tokens,
                &features[offset * 2..end * 2],
            )?);
        }
        Ok(labels)
    }

    fn punctuation_labels(&self, raw_tokens: &[String], final_input: bool) -> Result<Vec<usize>> {
        let content_limit = PunctuationModel::MAX_LENGTH - 2;
        let mut token_labels = Vec::with_capacity(raw_tokens.len());
        let mut chunk = Vec::new();
        let mut piece_count = 0;
        for token in raw_tokens {
            let token_piece_count = self.punctuation.wordpiece(token).len();
            if token_piece_count > content_limit {
                bail!("one ASR token exceeds the punctuation model limit")
            }
            if !chunk.is_empty() && piece_count + token_piece_count > content_limit {
                token_labels.extend(self.punctuation.predict(&chunk, false)?);
                chunk.clear();
                piece_count = 0;
            }
            chunk.push(token.clone());
            piece_count += token_piece_count;
        }
        if !chunk.is_empty() {
            token_labels.extend(self.punctuation.predict(&chunk, final_input)?);
        }
        let mut character_labels = Vec::new();
        for (token, label) in raw_tokens.iter().zip(token_labels) {
            character_labels.extend(std::iter::repeat_n(0, token.chars().count().saturating_sub(1)));
            character_labels.push(label);
        }
        Ok(character_labels)
    }
}

fn token_boundaries(tokens: &[String]) -> HashSet<usize> {
    let mut cursor = 0;
    tokens
        .iter()
        .map(|token| {
            cursor += token.chars().count();
            cursor - 1
        })
        .collect()
}

fn punctuate(
    words: &[RenderedWord],
    labels: &[usize],
    token_boundaries: &HashSet<usize>,
    chinese_punctuation: bool,
    final_input: bool,
    source_length: usize,
) -> Vec<ProcessedWord> {
    let mut output = Vec::new();
    for (index, rendered) in words.iter().enumerate() {
        output.push(rendered.word.clone());
        if rendered.word.kind == WordKind::Punctuation
            || words.get(index + 1).is_some_and(|next| {
                next.word.kind == WordKind::Punctuation
                    && next.source_end == rendered.source_end
            })
            || rendered.source_end == 0
        {
            continue;
        }
        let character_index = rendered.source_end - 1;
        let label = labels.get(character_index).copied().unwrap_or(0);
        if !token_boundaries.contains(&character_index) || !(1..=4).contains(&label) {
            continue;
        }
        if !final_input && character_index + 1 == source_length {
            continue;
        }
        let mark = punctuation_mark(label, chinese_punctuation);
        output.push(ProcessedWord::new(mark, WordKind::Punctuation));
    }
    output
}

fn punctuation_mark(label: usize, chinese: bool) -> &'static str {
    match (label, chinese) {
        (1, true) => "\u{ff0c}",
        (1, false) => ", ",
        (2, true) => "\u{3002}",
        (2, false) => ".",
        (3, true) => "\u{ff01}",
        (3, false) => "!",
        (4, true) => "\u{ff1f}",
        (4, false) => "?",
        _ => "",
    }
}

fn should_suppress_final_punctuation(
    raw_text: &str,
    rendered_text: &str,
    numeric_labels: &[String],
    raw_tokens: &[String],
) -> bool {
    let categories = numeric_labels
        .iter()
        .filter(|label| label.len() == 2)
        .map(|label| label.as_bytes()[1] as char)
        .collect::<HashSet<_>>();
    let pure_decimal = raw_text.contains('点')
        && !raw_text.starts_with('负')
        && !has_prefix_symbol(raw_text)
        && categories.contains(&'C')
        && !categories.contains(&'D')
        && numeric_labels
            .iter()
            .all(|label| !matches!(label.as_str(), "O" | "E" | "<unk>"));
    let short_pure_decimal = pure_decimal && is_short_decimal(rendered_text);
    let short_integer = rendered_text.chars().count() <= 2
        && !rendered_text.is_empty()
        && rendered_text.chars().all(|character| character.is_ascii_digit())
        && raw_text.chars().any(is_digit_character)
        && !raw_text.contains('洞');
    let bare_digit_sequence = !raw_text.is_empty()
        && raw_text.chars().all(is_sequence_digit)
        && !raw_text.contains('洞');
    let bare_split_number = is_bare_split_number(rendered_text)
        && raw_text.chars().all(is_numeric_feature_character);
    let bare_time = rendered_text.contains(':')
        && (raw_text.contains('点') || raw_text.contains('时'))
        && !raw_text.contains('分')
        && raw_text.chars().all(numeric_source_character);
    let malformed_decimal = raw_text.contains('点')
        && !rendered_text.contains('.')
        && !rendered_text.contains(':')
        && raw_text.chars().all(numeric_source_character);
    short_pure_decimal
        || short_integer
        || bare_digit_sequence
        || bare_split_number
        || bare_time
        || malformed_decimal
        || raw_tokens.is_empty()
}

fn is_short_decimal(value: &str) -> bool {
    let Some((whole, fraction)) = value.split_once('.') else {
        return false;
    };
    (1..=2).contains(&whole.len())
        && !fraction.is_empty()
        && whole.chars().all(|character| character.is_ascii_digit())
        && fraction.chars().all(|character| character.is_ascii_digit())
}

fn is_bare_split_number(value: &str) -> bool {
    let groups = value.split(' ').collect::<Vec<_>>();
    groups.len() > 1
        && groups.iter().all(|group| {
            !group.is_empty() && group.chars().all(|character| character.is_ascii_digit())
        })
}

fn normalize_midnight_zero(value: &str) -> String {
    let mut output = String::new();
    let characters = value.chars().collect::<Vec<_>>();
    for (index, character) in characters.iter().copied().enumerate() {
        if character == '0'
            && characters.get(index + 1) == Some(&':')
            && index.checked_sub(1).and_then(|previous| characters.get(previous)).is_none_or(|previous| {
                !previous.is_ascii_digit()
            })
        {
            output.push('0');
        }
        output.push(character);
    }
    output
}

fn apply_word_options(words: &mut [ProcessedWord], options: &PostprocessOptions) {
    if !options.english_punctuation {
        return;
    }
    for word in words {
        word.text = word
            .text
            .replace('\u{ff0c}', ",")
            .replace('\u{3002}', ".")
            .replace('\u{ff01}', "!")
            .replace('\u{ff1f}', "?")
            .replace('\u{ff1a}', ":")
            .replace('\u{ff1b}', ";")
            .replace('\u{3001}', ",");
    }
}

fn apply_output_options(mut text: String, options: &PostprocessOptions) -> String {
    if options.english_punctuation {
        text = text
            .replace('\u{ff0c}', ",")
            .replace('\u{3002}', ".")
            .replace('\u{ff01}', "!")
            .replace('\u{ff1f}', "?")
            .replace('\u{ff1a}', ":")
            .replace('\u{ff1b}', ";")
            .replace('\u{3001}', ",");
    }
    text
}

fn joined_text(words: &[RenderedWord]) -> String {
    words.iter().map(|word| word.word.text.as_str()).collect()
}

fn joined_processed_text(words: &[ProcessedWord]) -> String {
    words.iter().map(|word| word.text.as_str()).collect()
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x9fff)
}
