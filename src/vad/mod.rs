mod audio;
mod backend;
mod config;
mod segment;
mod segmenter;

pub use config::VadConfig;
pub use backend::VadSegmenter;
pub use segment::SpeechSegment;
#[cfg(test)]
pub use segment::SegmentReason;

#[cfg(test)]
mod tests;
