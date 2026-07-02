use std::io::Read;
use std::time::{Duration, Instant};

use tracing_indicatif::span_ext::IndicatifSpanExt;

pub(super) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for next in UNITS.iter().skip(1) {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{} {}", bytes, unit)
    } else {
        format!("{:.1} {}", value, unit)
    }
}

pub(super) struct ProgressThrottle<'a> {
    span: &'a tracing::Span,
    pending: u64,
    last_update: Instant,
    interval: Duration,
}

impl<'a> ProgressThrottle<'a> {
    pub(super) fn new(span: &'a tracing::Span) -> Self {
        Self {
            span,
            pending: 0,
            last_update: Instant::now(),
            interval: Duration::from_millis(200),
        }
    }

    pub(super) fn add(&mut self, bytes: u64, message: Option<&str>) {
        self.pending = self.pending.saturating_add(bytes);
        if self.last_update.elapsed() >= self.interval {
            self.flush(message);
        }
    }

    pub(super) fn flush(&mut self, message: Option<&str>) {
        if self.pending > 0 {
            self.span.pb_inc(self.pending);
            self.pending = 0;
        }
        if let Some(message) = message {
            self.span.pb_set_message(message);
        }
        self.last_update = Instant::now();
    }
}

pub(super) struct ProgressRead<'a, R> {
    inner: R,
    progress: ProgressThrottle<'a>,
}

impl<'a, R> ProgressRead<'a, R> {
    pub(super) fn new(inner: R, span: &'a tracing::Span) -> Self {
        Self {
            inner,
            progress: ProgressThrottle::new(span),
        }
    }

    pub(super) fn flush(&mut self) {
        self.progress.flush(None);
    }
}

impl<R: Read> Read for ProgressRead<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        if read > 0 {
            self.progress.add(read as u64, None);
        }
        Ok(read)
    }
}
