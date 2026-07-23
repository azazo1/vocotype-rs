use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Result, bail};
use flate2::read::GzDecoder;

use crate::postprocess_types::{ProcessedWord, RenderedWord};

#[derive(Clone, Debug)]
pub(crate) struct ReplacementTable {
    replacements: Vec<(String, String)>,
}

impl ReplacementTable {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let content = read_text(path)?;
        let replacements = content
            .lines()
            .filter_map(|line| line.trim_end_matches('\r').split_once(':'))
            .filter(|(source, _)| !source.is_empty())
            .map(|(source, target)| (source.to_string(), target.to_string()))
            .collect();
        Ok(Self { replacements })
    }

    #[cfg(test)]
    pub(crate) fn from_pairs(replacements: &[(&str, &str)]) -> Self {
        Self {
            replacements: replacements
                .iter()
                .map(|(source, target)| ((*source).to_string(), (*target).to_string()))
                .collect(),
        }
    }

    pub(crate) fn apply_words(&self, words: &[RenderedWord]) -> Result<Vec<RenderedWord>> {
        let mut output = words.to_vec();
        let mut text = joined_text(&output);
        for (source, target) in &self.replacements {
            if !text.contains(source) {
                continue;
            }
            let mut search_start = 0;
            while let Some(start) = find_source(&text, source, search_start) {
                let end = start + source.len();
                output = replace_once(&output, start, end, target)?;
                text.replace_range(start..end, target);
                search_start = start + target.len();
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NumericNotChangeList {
    phrases: Vec<String>,
}

impl NumericNotChangeList {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let content = read_text(path)?;
        let mut phrases = content
            .lines()
            .map(|line| line.trim_end_matches('\r'))
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        phrases.sort_unstable_by_key(|phrase| std::cmp::Reverse(phrase.chars().count()));
        phrases.dedup();
        Ok(Self { phrases })
    }

    pub(crate) fn protect(&self, text: &str, labels: &mut [String]) {
        for phrase in &self.phrases {
            let mut search_start = 0;
            while let Some(relative) = text[search_start..].find(phrase) {
                let byte_start = search_start + relative;
                let label_start = text[..byte_start].chars().count();
                let label_end = label_start + phrase.chars().count();
                for label in labels.get_mut(label_start..label_end).into_iter().flatten() {
                    *label = "O".to_string();
                }
                search_start = byte_start + phrase.len();
            }
        }
    }
}

fn read_text(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut content = String::new();
    if path.extension().is_some_and(|extension| extension == "gz") {
        GzDecoder::new(file).read_to_string(&mut content)?;
    } else {
        BufReader::new(file).read_to_string(&mut content)?;
    }
    Ok(content)
}

fn joined_text(words: &[RenderedWord]) -> String {
    words.iter().map(|word| word.word.text.as_str()).collect()
}

fn ascii_word_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn find_source(text: &str, source: &str, start: usize) -> Option<usize> {
    let mut cursor = start;
    while cursor <= text.len() {
        let relative = text.get(cursor..)?.find(source)?;
        let found = cursor + relative;
        let end = found + source.len();
        let left_ok = source
            .chars()
            .next()
            .is_none_or(|first| {
                !ascii_word_character(first)
                    || found == 0
                    || text[..found]
                        .chars()
                        .next_back()
                        .is_none_or(|previous| !ascii_word_character(previous))
            });
        let right_ok = source
            .chars()
            .next_back()
            .is_none_or(|last| {
                !ascii_word_character(last)
                    || end == text.len()
                    || text[end..]
                        .chars()
                        .next()
                        .is_none_or(|next| !ascii_word_character(next))
            });
        if left_ok && right_ok {
            return Some(found);
        }
        cursor = found + text[found..].chars().next()?.len_utf8();
    }
    None
}

fn replace_once(
    words: &[RenderedWord],
    start: usize,
    end: usize,
    target: &str,
) -> Result<Vec<RenderedWord>> {
    if start >= end || end > joined_text(words).len() {
        bail!("replacement range is invalid")
    }
    let mut offsets = Vec::with_capacity(words.len());
    let mut cursor = 0;
    for rendered in words {
        let next = cursor + rendered.word.text.len();
        offsets.push((cursor, next));
        cursor = next;
    }
    let first_index = offsets
        .iter()
        .position(|(_, finish)| *finish > start)
        .ok_or_else(|| anyhow::anyhow!("replacement start is outside rendered words"))?;
    let last_index = offsets
        .iter()
        .rposition(|(begin, _)| *begin < end)
        .ok_or_else(|| anyhow::anyhow!("replacement end is outside rendered words"))?;
    let first = &words[first_index];
    let last = &words[last_index];
    let prefix = &first.word.text[..start - offsets[first_index].0];
    let suffix = &last.word.text[end - offsets[last_index].0..];
    let source_begin = first_index
        .checked_sub(1)
        .map_or(0, |index| words[index].source_end);
    let prefix_source_end = source_begin + prefix.chars().count();
    let replacement_source_end = prefix_source_end.max(last.source_end - suffix.chars().count());

    let mut output = words[..first_index].to_vec();
    if !prefix.is_empty() {
        output.push(RenderedWord {
            word: ProcessedWord::new(prefix, first.word.kind),
            source_end: prefix_source_end,
        });
    }
    if !target.is_empty() {
        output.push(RenderedWord::new(
            target,
            Default::default(),
            replacement_source_end,
        ));
    }
    if !suffix.is_empty() {
        output.push(RenderedWord {
            word: ProcessedWord::new(suffix, last.word.kind),
            source_end: last.source_end,
        });
    }
    output.extend_from_slice(&words[last_index + 1..]);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess_types::WordKind;

    #[test]
    fn replacement_order_preserves_chaining() {
        let table = ReplacementTable::from_pairs(&[("a", "b"), ("b", "c"), ("d", "a")]);
        let words = vec![RenderedWord::new("a", WordKind::Text, 1)];
        let replaced = table.apply_words(&words).unwrap();
        assert_eq!(joined_text(&replaced), "c");
    }

    #[test]
    fn replacement_respects_ascii_word_boundaries() {
        let table = ReplacementTable::from_pairs(&[("cat", "dog")]);
        let words = vec![RenderedWord::new("copycat cat", WordKind::Text, 11)];
        let replaced = table.apply_words(&words).unwrap();
        assert_eq!(joined_text(&replaced), "copycat dog");
    }
}
