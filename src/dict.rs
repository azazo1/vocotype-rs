use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::warn;

use crate::config::CONFIG_DIR_NAME;

pub const DICT_FILE_NAME: &str = "dict.toml";
pub const DEFAULT_HOTWORDS_SCORE: f32 = 3.0;
const DEFAULT_MAX_RIME_WORDS: usize = 20_000;

const BUILTIN_HOTWORDS: &[&str] = &[
    "GPT", "ChatGPT", "OpenAI", "API", "URL", "HTTP", "HTTPS", "JSON", "SQL", "CPU", "GPU",
    "CLI", "UI", "ID",
];

const BUILTIN_REWRITES: &[(&str, &str)] = &[
    ("g p t", "GPT"),
    ("a p i", "API"),
    ("u r l", "URL"),
    ("h t t p", "HTTP"),
    ("h t t p s", "HTTPS"),
    ("j s o n", "JSON"),
    ("s q l", "SQL"),
    ("c p u", "CPU"),
    ("g p u", "GPU"),
    ("c l i", "CLI"),
    ("u i", "UI"),
    ("i d", "ID"),
];

#[derive(Clone, Debug)]
pub struct SpeechDictionary {
    hotwords: Vec<String>,
    hotword_rewrites: Vec<HotwordRewrite>,
    rewrites: Vec<RewriteRule>,
}

#[derive(Clone, Debug)]
pub struct LoadedDictionary {
    pub path: PathBuf,
    pub found: bool,
    pub rime_words: usize,
    pub dictionary: SpeechDictionary,
}

#[derive(Clone, Debug)]
struct RewriteRule {
    from: String,
    to: String,
}

#[derive(Clone, Debug)]
struct HotwordRewrite {
    from: String,
    to: String,
}

impl HotwordRewrite {
    fn new(word: &str) -> Option<Self> {
        let from = compact_hotword(word);
        if from.len() < 2 || !should_normalize_hotword(word) {
            return None;
        }
        Some(Self {
            from,
            to: word.to_string(),
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DictConfig {
    hotwords: Vec<String>,
    #[serde(alias = "rime-imports")]
    rime_imports: Vec<String>,
    #[serde(alias = "max-rime-words")]
    max_rime_words: Option<usize>,
    rewrites: HashMap<String, String>,
}

impl SpeechDictionary {
    pub fn builtin() -> Self {
        Self::from_parts(BUILTIN_HOTWORDS.iter().copied(), HashMap::new())
    }

    pub fn load(path: Option<&Path>) -> Result<Self> {
        Ok(Self::load_with_metadata(path)?.dictionary)
    }

    pub fn load_with_metadata(path: Option<&Path>) -> Result<LoadedDictionary> {
        let explicit_path = path.is_some();
        let path = path
            .map(expand_user_path)
            .unwrap_or_else(default_dict_path);
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !explicit_path => {
                return Ok(LoadedDictionary {
                    path,
                    found: false,
                    rime_words: 0,
                    dictionary: Self::builtin(),
                });
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法读取词汇表: {}", path.display()));
            }
        };
        let config = Self::config_from_toml(&content)
            .with_context(|| format!("无法解析词汇表: {}", path.display()))?;
        let (dictionary, rime_words) = Self::from_config(&config, &path)?;
        Ok(LoadedDictionary {
            path,
            found: true,
            rime_words,
            dictionary,
        })
    }

    fn from_config(config: &DictConfig, dict_path: &Path) -> Result<(Self, usize)> {
        let mut rime_words = Vec::new();
        let limit = config.max_rime_words.unwrap_or(DEFAULT_MAX_RIME_WORDS);
        for import in &config.rime_imports {
            let import_path = expand_user_path(Path::new(import));
            let import_path = if import_path.is_relative() {
                dict_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(import_path)
            } else {
                import_path
            };
            read_rime_words(
                &import_path,
                limit,
                &mut rime_words,
                &mut HashSet::new(),
            )?;
            if rime_words.len() >= limit {
                break;
            }
        }
        let rime_count = rime_words.len();
        let dictionary = Self::from_parts(
            config
                .hotwords
                .iter()
                .map(String::as_str)
                .chain(rime_words.iter().map(String::as_str)),
            config.rewrites.clone(),
        );
        Ok((dictionary, rime_count))
    }

    fn from_parts<'a>(
        hotwords: impl Iterator<Item = &'a str>,
        mut rewrites: HashMap<String, String>,
    ) -> Self {
        let mut seen = HashSet::new();
        let mut merged_hotwords = Vec::new();
        for word in BUILTIN_HOTWORDS.iter().copied().chain(hotwords) {
            let word = word.trim();
            if is_valid_hotword(word) && seen.insert(word.to_ascii_lowercase()) {
                merged_hotwords.push(word.to_string());
            }
        }

        for (from, to) in BUILTIN_REWRITES {
            rewrites
                .entry((*from).to_string())
                .or_insert_with(|| (*to).to_string());
        }
        let mut rewrite_rules = rewrites
            .into_iter()
            .filter_map(|(from, to)| {
                let from = from.trim();
                let to = to.trim();
                (!from.is_empty() && !to.is_empty()).then(|| RewriteRule {
                    from: from.to_string(),
                    to: to.to_string(),
                })
            })
            .collect::<Vec<_>>();
        rewrite_rules.sort_by_key(|rule| std::cmp::Reverse(rule.from.len()));
        let mut seen_hotword_rewrites = HashSet::new();
        let mut hotword_rewrites = merged_hotwords
            .iter()
            .filter_map(|word| HotwordRewrite::new(word))
            .filter(|rule| seen_hotword_rewrites.insert(rule.from.clone()))
            .collect::<Vec<_>>();
        hotword_rewrites.sort_by_key(|rule| std::cmp::Reverse(rule.from.len()));

        Self {
            hotwords: merged_hotwords,
            hotword_rewrites,
            rewrites: rewrite_rules,
        }
    }

    pub fn hotword_count(&self) -> usize {
        self.hotwords.len()
    }

    pub fn hotword_rewrite_count(&self) -> usize {
        self.hotword_rewrites.len()
    }

    pub fn rewrite_count(&self) -> usize {
        self.rewrites.len()
    }

    pub fn rewrite_text(&self, text: &str) -> String {
        let mut output = text.to_string();
        for rule in &self.rewrites {
            output = replace_ascii_case_insensitive(&output, &rule.from, &rule.to);
        }
        for rule in &self.hotword_rewrites {
            output = replace_compact_ascii_case_insensitive(&output, &rule.from, &rule.to);
        }
        output
    }

    fn config_from_toml(content: &str) -> Result<DictConfig> {
        toml::from_str(content).context("无法解析 TOML 词汇表")
    }
}

pub fn default_dict_path() -> PathBuf {
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().join(".config"))
        .unwrap_or_else(std::env::temp_dir)
        .join(CONFIG_DIR_NAME)
        .join(DICT_FILE_NAME)
}

pub fn default_dict_template() -> &'static str {
    r#"# VocoType 词汇表.
# 保存为 ~/.config/vocotype/dict.toml.

hotwords = [
  "GPT",
  "ChatGPT",
  "OpenAI",
  "API",
  "URL",
]

# sherpa-onnx Paraformer 模型不支持 sherpa contextual biasing.
# hotwords 用于后处理大小写和拆字母归一化.

# 仅导入英文 Rime 词条.
# Rime .dict.yaml 路径示例:
# macOS: "~/Library/Rime/wanxiang_english.dict.yaml"
# Linux: "~/.local/share/fcitx5/rime/wanxiang_english.dict.yaml"
# Linux: "~/.config/ibus/rime/wanxiang_english.dict.yaml"
# Windows: "C:/Users/Alice/AppData/Roaming/Rime/wanxiang_english.dict.yaml"
rime-imports = ["~/Library/Rime/wanxiang_english.dict.yaml"]
max-rime-words = 20000

[rewrites]
"g p t" = "GPT"
"a p i" = "API"
"u r l" = "URL"
"h t t p" = "HTTP"
"h t t p s" = "HTTPS"
"j s o n" = "JSON"
"s q l" = "SQL"
"#
}

pub fn write_dict_doctor(path: Option<&Path>, mut writer: impl Write) -> Result<()> {
    let loaded = SpeechDictionary::load_with_metadata(path)?;
    writeln!(writer, "词汇表检查")?;
    writeln!(writer, "路径: {}", loaded.path.display())?;
    if loaded.found {
        writeln!(writer, "状态: 已加载")?;
    } else {
        writeln!(writer, "状态: 未找到, 当前会使用内置词汇表")?;
    }
    writeln!(writer, "hotwords: {}", loaded.dictionary.hotword_count())?;
    writeln!(writer, "rewrites: {}", loaded.dictionary.rewrite_count())?;
    writeln!(writer, "rime-imported: {}", loaded.rime_words)?;
    Ok(())
}

fn read_rime_words(
    path: &Path,
    limit: usize,
    output: &mut Vec<String>,
    visited: &mut HashSet<PathBuf>,
) -> Result<()> {
    if output.len() >= limit {
        return Ok(());
    }
    let path = normalize_rime_path(path);
    if !visited.insert(path.clone()) {
        return Ok(());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("无法读取 Rime 词库: {}", path.display()))?;
    let imports = rime_import_tables(&content);
    for import in imports {
        let import_path = resolve_rime_import(&path, &import);
        if let Err(error) = read_rime_words(&import_path, limit, output, visited) {
            warn!(%error, path = %import_path.display(), "跳过 Rime 导入词库");
        }
        if output.len() >= limit {
            return Ok(());
        }
    }

    let mut in_entries = false;
    let mut seen = output
        .iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "..." {
            in_entries = true;
            continue;
        }
        if !in_entries || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let word = trimmed.split('\t').next().unwrap_or("").trim();
        if is_rime_english_word(word) && seen.insert(word.to_ascii_lowercase()) {
            output.push(word.to_string());
            if output.len() >= limit {
                break;
            }
        }
    }
    Ok(())
}

fn rime_import_tables(content: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut in_imports = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "..." {
            break;
        }
        if trimmed == "import_tables:" {
            in_imports = true;
            continue;
        }
        if in_imports {
            if let Some(value) = trimmed.strip_prefix("- ") {
                output.push(value.split('#').next().unwrap_or("").trim().to_string());
                continue;
            }
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                in_imports = false;
            }
        }
    }
    output.retain(|value| !value.is_empty());
    output
}

fn resolve_rime_import(base: &Path, import: &str) -> PathBuf {
    let mut path = PathBuf::from(import);
    if path.is_relative() {
        path = base.parent().unwrap_or_else(|| Path::new(".")).join(path);
    }
    normalize_rime_path(&path)
}

fn normalize_rime_path(path: &Path) -> PathBuf {
    if path.extension().is_none() {
        path.with_extension("dict.yaml")
    } else {
        path.to_path_buf()
    }
}

fn expand_user_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" {
        return directories::BaseDirs::new()
            .map(|dirs| dirs.home_dir().to_path_buf())
            .unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(dirs) = directories::BaseDirs::new()
    {
        return dirs.home_dir().join(rest);
    }
    path.to_path_buf()
}

fn is_valid_hotword(word: &str) -> bool {
    !word.is_empty()
        && word.len() <= 64
        && word.is_ascii()
        && word.chars().any(|ch| ch.is_ascii_alphabetic())
        && word
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '+' | '#' | '.' | '-' | '_' | '/'))
}

fn is_rime_english_word(word: &str) -> bool {
    is_valid_hotword(word)
}

fn should_normalize_hotword(word: &str) -> bool {
    word.chars().any(|ch| ch.is_ascii_uppercase())
        || word
            .chars()
            .any(|ch| matches!(ch, ' ' | '+' | '#' | '.' | '-' | '_' | '/'))
}

fn compact_hotword(word: &str) -> String {
    word.chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn replace_ascii_case_insensitive(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return text.to_string();
    }
    let lower_text = text.to_ascii_lowercase();
    let lower_from = from.to_ascii_lowercase();
    let mut output = String::new();
    let mut cursor = 0_usize;
    while let Some(relative) = lower_text[cursor..].find(&lower_from) {
        let start = cursor + relative;
        let end = start + lower_from.len();
        if is_rewrite_boundary(text, start, end) {
            output.push_str(&text[cursor..start]);
            output.push_str(to);
            cursor = end;
        } else {
            output.push_str(&text[cursor..end]);
            cursor = end;
        }
    }
    output.push_str(&text[cursor..]);
    output
}

fn is_rewrite_boundary(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !before.is_some_and(|ch| ch.is_ascii_alphanumeric())
        && !after.is_some_and(|ch| ch.is_ascii_alphanumeric())
}

fn replace_compact_ascii_case_insensitive(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return text.to_string();
    }
    let mut output = String::new();
    let mut cursor = 0_usize;
    while let Some((start, end)) = find_compact_ascii_match(text, cursor, from) {
        output.push_str(&text[cursor..start]);
        output.push_str(to);
        cursor = end;
    }
    if cursor == 0 {
        text.to_string()
    } else {
        output.push_str(&text[cursor..]);
        output
    }
}

fn find_compact_ascii_match(text: &str, cursor: usize, from: &str) -> Option<(usize, usize)> {
    let first = from.chars().next()?;
    for (relative, ch) in text[cursor..].char_indices() {
        if !ch.is_ascii() || ch.is_ascii_whitespace() {
            continue;
        }
        let start = cursor + relative;
        if ch.to_ascii_lowercase() != first {
            continue;
        }
        if let Some(end) = compact_ascii_match_end(text, start, from)
            && is_rewrite_boundary(text, start, end)
        {
            return Some((start, end));
        }
    }
    None
}

fn compact_ascii_match_end(text: &str, start: usize, from: &str) -> Option<usize> {
    let mut expected = from.chars();
    let mut current = expected.next()?;
    let mut matched = false;
    for (relative, ch) in text[start..].char_indices() {
        if ch.is_ascii_whitespace() {
            if matched {
                continue;
            }
            return None;
        }
        if !ch.is_ascii() || ch.to_ascii_lowercase() != current {
            return None;
        }
        matched = true;
        let end = start + relative + ch.len_utf8();
        match expected.next() {
            Some(next) => current = next,
            None => return Some(end),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_spelled_letters() {
        let dict = SpeechDictionary::builtin();
        assert_eq!(dict.rewrite_text("g p t"), "GPT");
        assert_eq!(dict.rewrite_text("使用 g p t 写代码"), "使用 GPT 写代码");
        assert_eq!(dict.rewrite_text("G P T"), "GPT");
    }

    #[test]
    fn user_rewrite_overrides_builtin() {
        let config = SpeechDictionary::config_from_toml(
            r#"
[rewrites]
"g p t" = "ChatGPT"
"#,
        )
        .unwrap();
        let (dict, _) = SpeechDictionary::from_config(&config, Path::new("/tmp/dict.toml")).unwrap();
        assert_eq!(dict.rewrite_text("g p t"), "ChatGPT");
    }

    #[test]
    fn hotwords_are_normalized_after_rewrites() {
        let dict = SpeechDictionary::builtin();
        assert_eq!(dict.rewrite_text("g p t"), "GPT");
        assert_eq!(dict.rewrite_text("G P T"), "GPT");
        assert_eq!(dict.rewrite_text("we use gpt"), "we use GPT");
        assert_eq!(dict.rewrite_text("chat g p t"), "ChatGPT");
        assert_eq!(dict.rewrite_text("open ai"), "OpenAI");
        assert_eq!(dict.rewrite_text("xxgpt"), "xxgpt");
    }

    #[test]
    fn duplicate_compact_hotwords_keep_first_surface() {
        let config = SpeechDictionary::config_from_toml(
            r#"
hotwords = [
  "g p t",
]
"#,
        )
        .unwrap();
        let (dict, _) = SpeechDictionary::from_config(&config, Path::new("/tmp/dict.toml")).unwrap();
        assert_eq!(dict.rewrite_text("gpt"), "GPT");
    }

    #[test]
    fn parses_rime_import_tables() {
        let imports = rime_import_tables(
            r#"
---
name: test
import_tables:
  - dicts/en
  - custom/terms # comment
...
"#,
        );
        assert_eq!(imports, vec!["dicts/en", "custom/terms"]);
    }

    #[test]
    fn filters_rime_words_to_english() {
        assert!(is_rime_english_word("GPT"));
        assert!(is_rime_english_word("ChatGPT"));
        assert!(is_rime_english_word("gpt"));
        assert!(!is_rime_english_word("中文"));
    }
}
