use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

pub const CONFIG_DIR_NAME: &str = "vocotype";
pub const CONFIG_FILE_NAME: &str = "config.toml";

pub fn default_config_template() -> &'static str {
    r#"# VocoType 配置文件.
# 保存为 ~/.config/vocotype/config.toml.

# model-dir = "/path/to/models"
# model-cache-dir = "/path/to/model-cache"
# dict-path = "~/.config/vocotype/dict.toml"
model-revision = "asr-models"

[asr]
hotwords-score = 3.0

[post-processing]
english-punctuation = false
strip-trailing-period = false

[daemon]
# 热键可以写单键或组合键. 组合键用 + 连接, 修饰键在前, 主键在后.
# 示例: "F2", "ctrl+f2", "cmdorctrl+space", "shift+alt+KeyQ".
hotkey = "F2"
# 可选值: "pressed", "toggle", "trigger-end".
hotkey-mode = "pressed"
# trigger-end 模式下需要配置结束热键.
# end-hotkey = "F3"
save-dataset = false
# dataset-dir = "/path/to/dataset"
append-newline = false
inject-method = "auto"
end-silence-ms = 650
pre-roll-ms = 180
tail-padding-ms = 180
min-speech-ms = 240
max-segment-ms = 15000
idle-unload-secs = 300

[transcribe]
format = "json"
pretty = false
subtitle-max-chars = 24
"#
}

pub fn config_schema() -> &'static str {
    r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://vocotype.local/schema/config.schema.json",
  "title": "VocoType 配置文件",
  "description": "VocoType 持久配置. 不包含单次输出文件这类临时参数.",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "model-dir": {
      "type": "string",
      "description": "模型加载根目录."
    },
    "model-cache-dir": {
      "type": "string",
      "description": "模型下载缓存目录."
    },
    "model-revision": {
      "type": "string",
      "default": "asr-models",
      "description": "模型 manifest 记录的 revision."
    },
    "dict-path": {
      "type": "string",
      "description": "用户词汇表 dict.toml 路径."
    },
    "asr": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "hotwords-score": {
          "type": "number",
          "minimum": 0,
          "default": 3.0,
          "description": "sherpa-onnx hotwords 加权分数."
        }
      }
    },
    "post-processing": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "english-punctuation": {
          "type": "boolean",
          "default": false,
          "description": "是否把最终转写文本中的中文标点转换为 ASCII 标点."
        },
        "strip-trailing-period": {
          "type": "boolean",
          "default": false,
          "description": "是否删除最终转写文本末尾的句号."
        }
      }
    },
    "daemon": {
      "type": "object",
      "additionalProperties": false,
      "allOf": [
        {
          "if": {
            "properties": {
              "hotkey-mode": {
                "const": "trigger-end"
              }
            },
            "required": ["hotkey-mode"]
          },
          "then": {
            "required": ["end-hotkey"]
          }
        }
      ],
      "properties": {
        "hotkey": {
          "type": "string",
          "default": "F2",
          "description": "按住录音的全局热键. 支持单键或组合键. 组合键用 + 连接, 修饰键在前, 主键在后, 例如 F2, ctrl+f2, cmdorctrl+space, shift+alt+KeyQ.",
          "examples": ["F2", "ctrl+f2", "cmdorctrl+space", "shift+alt+KeyQ"]
        },
        "hotkey-mode": {
          "type": "string",
          "enum": ["pressed", "toggle", "trigger-end"],
          "default": "pressed",
          "description": "热键触发模式. pressed 表示按住 hotkey 录音并在松开时停止. toggle 表示按一次 hotkey 开始, 再按一次停止. trigger-end 表示按 hotkey 开始, 按 end-hotkey 停止."
        },
        "end-hotkey": {
          "type": "string",
          "description": "trigger-end 模式下用于停止录音的结束热键. 写法和 hotkey 相同.",
          "examples": ["F3", "ctrl+f3", "cmdorctrl+space"]
        },
        "save-dataset": {
          "type": "boolean",
          "default": false,
          "description": "是否保存转写音频和结果用于数据集."
        },
        "dataset-dir": {
          "type": "string",
          "description": "数据集保存目录."
        },
        "append-newline": {
          "type": "boolean",
          "default": false,
          "description": "注入文本后是否追加换行."
        },
        "inject-method": {
          "type": "string",
          "enum": ["auto", "type", "clipboard"],
          "default": "auto",
          "description": "文本注入方式."
        },
        "end-silence-ms": {
          "type": "integer",
          "minimum": 0,
          "default": 650,
          "description": "判定语音结束的静音毫秒数."
        },
        "pre-roll-ms": {
          "type": "integer",
          "minimum": 0,
          "default": 180,
          "description": "分段前保留的音频毫秒数."
        },
        "tail-padding-ms": {
          "type": "integer",
          "minimum": 0,
          "default": 180,
          "description": "分段后追加的音频毫秒数."
        },
        "min-speech-ms": {
          "type": "integer",
          "minimum": 0,
          "default": 240,
          "description": "保留语音段的最短毫秒数."
        },
        "max-segment-ms": {
          "type": "integer",
          "minimum": 1,
          "default": 15000,
          "description": "单个语音段的最长毫秒数."
        },
        "idle-unload-secs": {
          "type": "integer",
          "minimum": 0,
          "default": 300,
          "description": "daemon 队列空闲多少秒后卸载 ASR 和 PUNC 模型. 0 表示不自动卸载."
        }
      }
    },
    "transcribe": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "format": {
          "type": "string",
          "enum": ["json", "srt"],
          "default": "json",
          "description": "默认输出格式."
        },
        "pretty": {
          "type": "boolean",
          "default": false,
          "description": "JSON 输出是否使用 pretty 格式."
        },
        "subtitle-max-chars": {
          "type": "integer",
          "minimum": 1,
          "default": 24,
          "description": "SRT 每条字幕的目标最大字符数."
        }
      }
    }
  }
}
"#
}

#[derive(Clone, Debug)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub found: bool,
    pub config: AppConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    #[serde(alias = "model-dir")]
    pub model_dir: Option<PathBuf>,
    #[serde(alias = "model-cache-dir")]
    pub model_cache_dir: Option<PathBuf>,
    #[serde(alias = "model-revision")]
    pub model_revision: Option<String>,
    #[serde(alias = "dict-path")]
    pub dict_path: Option<PathBuf>,
    pub asr: AsrConfig,
    #[serde(alias = "post-processing")]
    pub post_processing: PostProcessingConfig,
    pub daemon: DaemonConfig,
    pub transcribe: TranscribeConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    #[serde(alias = "hotwords-score")]
    pub hotwords_score: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PostProcessingConfig {
    #[serde(alias = "english-punctuation")]
    pub english_punctuation: Option<bool>,
    #[serde(alias = "strip-trailing-period")]
    pub strip_trailing_period: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub hotkey: Option<String>,
    #[serde(alias = "hotkey-mode")]
    pub hotkey_mode: Option<String>,
    #[serde(alias = "end-hotkey")]
    pub end_hotkey: Option<String>,
    #[serde(alias = "save-dataset")]
    pub save_dataset: Option<bool>,
    #[serde(alias = "dataset-dir")]
    pub dataset_dir: Option<PathBuf>,
    #[serde(alias = "append-newline")]
    pub append_newline: Option<bool>,
    #[serde(alias = "inject-method")]
    pub inject_method: Option<String>,
    #[serde(alias = "end-silence-ms")]
    pub end_silence_ms: Option<u32>,
    #[serde(alias = "pre-roll-ms")]
    pub pre_roll_ms: Option<u32>,
    #[serde(alias = "tail-padding-ms")]
    pub tail_padding_ms: Option<u32>,
    #[serde(alias = "min-speech-ms")]
    pub min_speech_ms: Option<u32>,
    #[serde(alias = "max-segment-ms")]
    pub max_segment_ms: Option<u32>,
    #[serde(alias = "idle-unload-secs")]
    pub idle_unload_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranscribeConfig {
    pub format: Option<String>,
    pub pretty: Option<bool>,
    #[serde(alias = "subtitle-max-chars")]
    pub subtitle_max_chars: Option<usize>,
}

impl AppConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let loaded = Self::load_with_metadata(path)?;
        if loaded.found {
            info!(path = %loaded.path.display(), "已加载配置文件");
        }
        Ok(loaded.config)
    }

    pub fn load_with_metadata(path: Option<&Path>) -> Result<LoadedConfig> {
        let explicit_path = path.is_some();
        let path = path.map(Path::to_path_buf).unwrap_or_else(default_config_path);
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !explicit_path => {
                return Ok(LoadedConfig {
                    path,
                    found: false,
                    config: Self::default(),
                });
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法读取配置文件: {}", path.display()));
            }
        };
        let config = Self::from_toml(&content)
            .with_context(|| format!("无法解析配置文件: {}", path.display()))?;
        Ok(LoadedConfig {
            path,
            found: true,
            config,
        })
    }

    pub fn from_toml(content: &str) -> Result<Self> {
        toml::from_str(content).context("无法解析 TOML 配置")
    }
}

pub fn default_config_path() -> PathBuf {
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().join(".config"))
        .unwrap_or_else(std::env::temp_dir)
        .join(CONFIG_DIR_NAME)
        .join(CONFIG_FILE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_parses() {
        AppConfig::from_toml(default_config_template()).unwrap();
    }

    #[test]
    fn schema_is_valid_json() {
        serde_json::from_str::<serde_json::Value>(config_schema()).unwrap();
    }

    #[test]
    fn parses_kebab_case_keys() {
        let config = AppConfig::from_toml(
            r#"
model-dir = "/tmp/vocotype-models"
model-revision = "test-revision"
dict-path = "/tmp/vocotype-dict.toml"

[asr]
hotwords-score = 4.5

[post-processing]
english-punctuation = true
strip-trailing-period = true

[daemon]
idle-unload-secs = 42
tail-padding-ms = 250
hotkey-mode = "toggle"
end-hotkey = "F3"

[transcribe]
subtitle-max-chars = 32
"#,
        )
        .unwrap();

        assert_eq!(config.model_dir, Some(PathBuf::from("/tmp/vocotype-models")));
        assert_eq!(config.model_revision.as_deref(), Some("test-revision"));
        assert_eq!(
            config.dict_path,
            Some(PathBuf::from("/tmp/vocotype-dict.toml"))
        );
        assert_eq!(config.asr.hotwords_score, Some(4.5));
        assert_eq!(config.post_processing.english_punctuation, Some(true));
        assert_eq!(config.post_processing.strip_trailing_period, Some(true));
        assert_eq!(config.daemon.idle_unload_secs, Some(42));
        assert_eq!(config.daemon.tail_padding_ms, Some(250));
        assert_eq!(config.daemon.hotkey_mode.as_deref(), Some("toggle"));
        assert_eq!(config.daemon.end_hotkey.as_deref(), Some("F3"));
        assert_eq!(config.transcribe.subtitle_max_chars, Some(32));
    }

    #[test]
    fn rejects_legacy_daemon_strip_trailing_period() {
        let config = AppConfig::from_toml(
            r#"
[daemon]
strip-trailing-period = true
"#,
        );

        assert!(config.is_err());
    }

    #[test]
    fn parses_snake_case_keys() {
        let config = AppConfig::from_toml(
            r#"
model_dir = "/tmp/vocotype-models"
dict_path = "/tmp/vocotype-dict.toml"

[asr]
hotwords_score = 2.5

[post_processing]
english_punctuation = true
strip_trailing_period = true

[daemon]
save_dataset = true
dataset_dir = "/tmp/dataset"
"#,
        )
        .unwrap();

        assert_eq!(config.model_dir, Some(PathBuf::from("/tmp/vocotype-models")));
        assert_eq!(
            config.dict_path,
            Some(PathBuf::from("/tmp/vocotype-dict.toml"))
        );
        assert_eq!(config.asr.hotwords_score, Some(2.5));
        assert_eq!(config.post_processing.english_punctuation, Some(true));
        assert_eq!(config.post_processing.strip_trailing_period, Some(true));
        assert_eq!(config.daemon.save_dataset, Some(true));
        assert_eq!(config.daemon.dataset_dir, Some(PathBuf::from("/tmp/dataset")));
    }

    #[test]
    fn rejects_single_run_output_path() {
        let config = AppConfig::from_toml(
            r#"
[transcribe]
output = "/tmp/result.srt"
"#,
        );

        assert!(config.is_err());
    }
}
