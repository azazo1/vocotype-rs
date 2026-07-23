#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum WordKind {
    #[default]
    Text,
    Number,
    Punctuation,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProcessedWord {
    pub(crate) text: String,
    pub(crate) kind: WordKind,
}

impl ProcessedWord {
    pub(crate) fn new(text: impl Into<String>, kind: WordKind) -> Self {
        Self {
            text: text.into(),
            kind,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RenderedWord {
    pub(crate) word: ProcessedWord,
    pub(crate) source_end: usize,
}

impl RenderedWord {
    pub(crate) fn new(
        text: impl Into<String>,
        kind: WordKind,
        source_end: usize,
    ) -> Self {
        Self {
            word: ProcessedWord::new(text, kind),
            source_end,
        }
    }
}
