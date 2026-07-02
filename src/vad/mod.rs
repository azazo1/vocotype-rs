mod audio;
mod config;
mod segment;
mod segmenter;

pub use config::VadConfig;
pub use segment::SpeechSegment;
#[cfg(test)]
pub use segment::SegmentReason;
pub use segmenter::VadSegmenter;

#[cfg(test)]
mod tests;
