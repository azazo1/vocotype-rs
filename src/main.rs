mod app;
mod asr;
mod asr_backend;
mod audio;
mod cli;
mod config;
mod daemon;
mod dataset;
mod dict;
mod hotkey;
mod inject;
mod logging;
mod models;
mod overlay;
mod punctuation;
mod subtitle;
mod vad;
mod wav;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    logging::init()?;
    cli::run().await
}
