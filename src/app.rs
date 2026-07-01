use std::path::{Path, PathBuf};

pub const APP_DIR_NAME: &str = "vocotype-rs";

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub model_root: PathBuf,
    pub model_cache_root: PathBuf,
    pub log_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl AppPaths {
    pub fn resolve(model_root: Option<PathBuf>, model_cache_root: Option<PathBuf>) -> Self {
        let base_dirs = directories::BaseDirs::new();
        let data_base = base_dirs
            .as_ref()
            .map(|dirs| dirs.data_local_dir().join(APP_DIR_NAME))
            .unwrap_or_else(|| std::env::temp_dir().join(APP_DIR_NAME));
        let cache_base = base_dirs
            .as_ref()
            .map(|dirs| dirs.cache_dir().join(APP_DIR_NAME))
            .unwrap_or_else(|| std::env::temp_dir().join(APP_DIR_NAME));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_dir_uses_app_folder_name() {
        let paths = AppPaths::resolve(None, None);
        assert_eq!(paths.model_cache_root.file_name().unwrap(), "models");
        assert_eq!(
            paths
                .model_cache_root
                .parent()
                .unwrap()
                .file_name()
                .unwrap(),
            APP_DIR_NAME
        );
    }

    #[test]
    fn default_data_dir_uses_app_folder_name() {
        let paths = AppPaths::resolve(None, None);
        assert_eq!(paths.data_dir.file_name().unwrap(), APP_DIR_NAME);
        assert_eq!(paths.log_dir, paths.data_dir.join("logs"));
    }

    #[test]
    fn explicit_cache_dir_overrides_default() {
        let paths = AppPaths::resolve(None, Some(PathBuf::from("/tmp/custom-cache")));
        assert_eq!(paths.model_root, PathBuf::from("/tmp/custom-cache"));
        assert_eq!(paths.model_cache_root, PathBuf::from("/tmp/custom-cache"));
    }
}
