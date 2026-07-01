mod app;
mod audio;
mod asr;
mod cli;
mod dataset;
mod daemon;
mod hotkey;
mod inject;
mod logging;
mod models;
mod overlay;
mod vad;
mod wav;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    logging::init()?;
    cli::run().await
}
