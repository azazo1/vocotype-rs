use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub model_root: PathBuf,
    pub model_cache_root: PathBuf,
    pub log_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl AppPaths {
    pub fn resolve(model_root: Option<PathBuf>, model_cache_root: Option<PathBuf>) -> Self {
        let dirs = directories::ProjectDirs::from("com", "vocotype", "vocotype-rs");
        let data_base = dirs
            .as_ref()
            .map(|dirs| dirs.data_local_dir().to_path_buf())
            .unwrap_or_else(|| std::env::temp_dir().join("vocotype-rs"));
        let cache_base = dirs
            .as_ref()
            .map(|dirs| dirs.cache_dir().to_path_buf())
            .unwrap_or_else(|| std::env::temp_dir().join("vocotype-rs-cache"));

        let default_model_cache = cache_base.join("models");
        let default_model_root = model_cache_root
            .clone()
            .unwrap_or_else(|| default_model_cache.clone());

        Self {
            model_root: model_root.unwrap_or_else(|| default_model_root.clone()),
            model_cache_root: model_cache_root.unwrap_or(default_model_cache),
            log_dir: data_base.join("logs"),
            data_dir: data_base,
        }
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.model_root)?;
        std::fs::create_dir_all(&self.model_cache_root)?;
        std::fs::create_dir_all(&self.log_dir)?;
        std::fs::create_dir_all(&self.data_dir)?;
        Ok(())
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.model_root.join("manifest.json")
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.model_root.join(name)
    }

    pub fn cache_entry_dir(&self, name: &str) -> PathBuf {
        self.model_cache_root.join(name)
    }
}

pub fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

pub fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
