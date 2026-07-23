use std::fmt;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
pub enum AsrBackend {
    #[default]
    Sherpa,
    Iflytek,
}

impl fmt::Display for AsrBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Sherpa => "sherpa",
            Self::Iflytek => "iflytek",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_labels_are_stable() {
        assert_eq!(AsrBackend::Sherpa.to_string(), "sherpa");
        assert_eq!(AsrBackend::Iflytek.to_string(), "iflytek");
    }
}
