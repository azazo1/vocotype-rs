use anyhow::Result;
use tracing_indicatif::{
    IndicatifLayer,
    filter::{IndicatifFilter, hide_indicatif_span_fields},
};
use tracing_subscriber::fmt::format::DefaultFields;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub fn init() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let indicatif_layer = IndicatifLayer::new()
        .with_span_field_formatter(hide_indicatif_span_fields(DefaultFields::new()));
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(true)
        .with_timer(fmt::time::SystemTime)
        .with_writer(indicatif_layer.get_stderr_writer());

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(indicatif_layer.with_filter(IndicatifFilter::new(false)))
        .try_init()
        .ok();

    tracing::info!(
        target: "vocotype_rs::logging",
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        "日志系统已初始化"
    );
    Ok(())
}
