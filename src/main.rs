mod app;
mod asr;
mod audio;
mod cli;
mod daemon;
mod dataset;
mod hotkey;
mod inject;
mod logging;
mod models;
mod overlay;
mod subtitle;
mod vad;
mod wav;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    logging::init()?;
    cli::run().await
}
