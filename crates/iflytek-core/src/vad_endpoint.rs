use anyhow::{Result, bail};

const HISTORY_SIZE: usize = 256;
const START_WINDOW: i32 = 20;
const START_REQUIRED: i32 = 15;
const MAYBE_START_WINDOW: i32 = 7;

#[derive(Clone, Copy, Debug)]
pub struct VadEndpointConfig {
    pub energy_threshold: f32,
    pub frame_start_margin: i32,
    pub frame_end_margin: i32,
    pub end_gap: i32,
    pub pre_speech_end_on: bool,
    pub pre_speech_end: i32,
    pub time_two_pre_end: i32,
    pub delay_speech_end_on: bool,
    pub delay_speech_end_gap: i32,
    pub vad_threshold: f32,
    pub response_timeout: i32,
    pub speech_end: i32,
    pub force_segment: i32,
}

impl Default for VadEndpointConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.0,
            frame_start_margin: 30,
            frame_end_margin: 30,
            end_gap: 300,
            pre_speech_end_on: false,
            pre_speech_end: 50,
            time_two_pre_end: 100,
            delay_speech_end_on: false,
            delay_speech_end_gap: 100,
            vad_threshold: 0.0,
            response_timeout: 6_000,
            speech_end: 800,
            force_segment: 6_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum VadStatus {
    None = 0,
    SpeechStart = 1,
    Speech = 2,
    SpeechEnd = 3,
    Timeout = 4,
    PreSpeechEnd = 5,
    DelayedSpeechEnd = 6,
}

#[derive(Clone, Copy, Debug)]
pub struct VadEvidenceFrame {
    pub frame: i32,
    pub speech_probability: f32,
    pub silence_probability: f32,
    pub neural_speech: bool,
    pub energy_active: bool,
    pub speech_evidence: bool,
    pub cumulative_speech: i32,
    pub silence_run: i32,
}

#[derive(Clone, Debug)]
pub struct VadEndpointState {
    cumulative_speech: [i32; HISTORY_SIZE],
    silence_history: [i32; HISTORY_SIZE],
    pub start_pause_frame: i32,
    pub may_pause_frame: i32,
    pub end_pause_frame: i32,
    pub current_frame: i32,
    pub checked_frame_count: i32,
    pub read_delay: i32,
    pub frame_margin: i32,
    pub pre_end_pause_frame: i32,
    pub real_pause_frame: i32,
    pub silence_run: i32,
    pub last_silence_frame: i32,
    pub segment_count: i32,
    pub finished: bool,
    pub response_timeout: bool,
    pub speech_timeout: bool,
    pub delay_end_found: bool,
    pub pre_end_status: i32,
    pub current_status: i32,
}

impl Default for VadEndpointState {
    fn default() -> Self {
        Self {
            cumulative_speech: [0; HISTORY_SIZE],
            silence_history: [0; HISTORY_SIZE],
            start_pause_frame: -1,
            may_pause_frame: -1,
            end_pause_frame: -1,
            current_frame: 0,
            checked_frame_count: 0,
            read_delay: 0,
            frame_margin: -1,
            pre_end_pause_frame: -1,
            real_pause_frame: -1,
            silence_run: 0,
            last_silence_frame: 0,
            segment_count: 0,
            finished: false,
            response_timeout: false,
            speech_timeout: false,
            delay_end_found: false,
            pre_end_status: -1,
            current_status: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OriginalVadEndpoint {
    config: VadEndpointConfig,
    state: VadEndpointState,
}

impl OriginalVadEndpoint {
    pub fn new(config: VadEndpointConfig) -> Result<Self> {
        if config.frame_start_margin < 0
            || config.frame_end_margin < 0
            || config.end_gap < 0
            || config.pre_speech_end < 0
            || config.time_two_pre_end < 0
            || config.delay_speech_end_gap < 0
            || config.response_timeout < 0
            || config.speech_end < 0
            || config.force_segment < 0
            || !config.energy_threshold.is_finite()
            || !config.vad_threshold.is_finite()
            || !(0.0..=1.0).contains(&config.vad_threshold)
        {
            bail!("invalid EdgeEsr VAD endpoint configuration")
        }
        Ok(Self {
            config,
            state: VadEndpointState::default(),
        })
    }

    pub fn state(&self) -> &VadEndpointState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut VadEndpointState {
        &mut self.state
    }

    pub fn reset(&mut self) {
        self.state = VadEndpointState::default();
    }

    pub fn push_logits(
        &mut self,
        speech_logit: f32,
        silence_logit: f32,
        energy_active: bool,
    ) -> VadEvidenceFrame {
        let first = speech_logit.exp();
        let second = silence_logit.exp();
        let denominator = first + second;
        let speech_probability = first / denominator;
        let silence_probability = second / denominator;
        let threshold_term = self.config.vad_threshold * 2.0 - 1.0;
        let neural_speech = speech_probability - silence_probability - threshold_term >= 0.0;
        let speech_evidence = neural_speech && energy_active;
        self.push_evidence(
            speech_probability,
            silence_probability,
            neural_speech,
            energy_active,
            speech_evidence,
            true,
        )
    }

    pub fn pad_silence(&mut self) -> VadEvidenceFrame {
        self.push_evidence(f32::NAN, f32::NAN, false, false, false, false)
    }

    pub fn step(&mut self, frame: i32, flush_frame: i32) -> Result<VadStatus> {
        if frame < 0 || frame >= self.state.checked_frame_count {
            bail!("EdgeEsr VAD endpoint frame is not classified")
        }
        self.state.current_frame = frame;
        let pre_end_window = self.pre_end_window();
        let pre_end_timeout = self
            .config
            .time_two_pre_end
            .max(self.config.end_gap - pre_end_window);
        if pre_end_window < self.config.end_gap
            && self.state.pre_end_status > 0
            && self.state.start_pause_frame > 0
            && frame - self.state.pre_end_status > pre_end_timeout
        {
            self.state.pre_end_status = -1;
        }

        if self.state.start_pause_frame < 0 {
            let current = self.cumulative(frame);
            if current - self.cumulative((frame - START_WINDOW).max(0)) > START_REQUIRED - 1 {
                return Ok(self.start_segment(frame));
            }
            let mut may_pause = self.state.may_pause_frame;
            if may_pause < 0 || frame - may_pause > self.config.end_gap {
                self.state.may_pause_frame = -1;
                may_pause = -1;
            } else if current - self.cumulative((frame - 1).max(0)) == 1
                && current - self.cumulative(may_pause) > START_REQUIRED - 1
            {
                return Ok(self.start_segment(may_pause + START_WINDOW));
            }
            if self.state.segment_count == 0 && frame > self.config.response_timeout {
                self.state.response_timeout = true;
                self.state.read_delay = 0;
                self.state.pre_end_status = -1;
                self.state.delay_end_found = false;
                return Ok(VadStatus::Timeout);
            }
            if self.state.segment_count != 0 {
                if self.config.delay_speech_end_on
                    && !self.state.delay_end_found
                    && self.state.checked_frame_count - self.state.last_silence_frame
                        > self.config.delay_speech_end_gap
                {
                    self.state.read_delay = 0;
                    self.state.delay_end_found = true;
                    return Ok(VadStatus::DelayedSpeechEnd);
                }
                if self.state.checked_frame_count - self.state.last_silence_frame
                    > self.config.speech_end
                {
                    self.state.speech_timeout = true;
                    self.state.delay_end_found = false;
                    self.state.read_delay = 0;
                    self.state.pre_end_status = -1;
                    return Ok(VadStatus::Timeout);
                }
            }
            if may_pause < 0
                && self.speech_count(frame, MAYBE_START_WINDOW) > MAYBE_START_WINDOW - 1
            {
                self.state.may_pause_frame = frame - MAYBE_START_WINDOW;
            }
            if self.state.segment_count != 0
                && let Some(status) = self.check_pre_end(frame, pre_end_window)
            {
                return Ok(status);
            }
            return Ok(VadStatus::None);
        }

        if let Some(status) = self.check_pre_end(frame, pre_end_window) {
            return Ok(status);
        }
        if frame > self.config.end_gap {
            let end_candidate = frame - self.config.end_gap;
            if self.cumulative(frame) == self.cumulative(end_candidate) {
                self.state.current_status = VadStatus::SpeechEnd as i32;
                let real_pause = end_candidate - self.silence_run(end_candidate);
                self.state.pre_end_pause_frame = real_pause + self.config.frame_end_margin;
                self.state.end_pause_frame = self.state.pre_end_pause_frame;
                self.state.real_pause_frame = real_pause;
                self.state.start_pause_frame = -1;
                self.state.frame_margin = -1;
                self.state.last_silence_frame = real_pause;
                return Ok(VadStatus::SpeechEnd);
            }
        }
        if flush_frame - 1 == frame {
            self.state.current_status = VadStatus::SpeechEnd as i32;
            if self.state.frame_margin > 0 {
                self.state.end_pause_frame = self.state.frame_margin;
                self.state.pre_end_pause_frame = self.state.frame_margin;
                self.state.real_pause_frame = self.state.frame_margin - self.config.frame_end_margin;
            } else {
                self.state.end_pause_frame = frame;
                self.state.pre_end_pause_frame = frame;
                self.state.real_pause_frame = frame;
            }
            self.state.start_pause_frame = -1;
            self.state.read_delay = i32::from(self.state.frame_margin < 0);
            self.state.frame_margin = -1;
            self.state.last_silence_frame = self.state.real_pause_frame;
            return Ok(VadStatus::SpeechEnd);
        }
        if self.state.frame_margin < 0 {
            if frame > self.config.frame_end_margin
                && self.cumulative(frame)
                    == self.cumulative(frame - self.config.frame_end_margin)
            {
                self.state.frame_margin = frame;
                self.state.read_delay = 0;
                return Ok(VadStatus::None);
            }
            self.state.read_delay = 1;
            return Ok(VadStatus::Speech);
        }
        if frame > 1 && self.cumulative(frame) == self.cumulative(frame - 1) {
            self.state.read_delay = 0;
            return Ok(VadStatus::None);
        }
        self.state.read_delay = frame + 1 - self.state.frame_margin;
        self.state.frame_margin = -1;
        Ok(VadStatus::Speech)
    }

    pub fn finalize(&mut self, frame: i32) -> VadStatus {
        if self.state.start_pause_frame < 1 {
            self.state.finished = true;
            return VadStatus::Timeout;
        }
        self.state.end_pause_frame = frame;
        self.state.real_pause_frame = frame;
        self.state.start_pause_frame = -1;
        VadStatus::SpeechEnd
    }

    fn push_evidence(
        &mut self,
        speech_probability: f32,
        silence_probability: f32,
        neural_speech: bool,
        energy_active: bool,
        speech_evidence: bool,
        update_scalar_silence: bool,
    ) -> VadEvidenceFrame {
        let frame = self.state.checked_frame_count;
        let previous_index = if frame <= 0 {
            0
        } else {
            (frame - 1) as usize % HISTORY_SIZE
        };
        let previous = self.state.cumulative_speech[previous_index];
        let mut contribution = 0;
        let silence_run = if update_scalar_silence {
            if speech_evidence {
                self.state.silence_run = 0;
                contribution = 1;
            } else {
                self.state.silence_run += 1;
            }
            self.state.silence_run
        } else {
            self.state.silence_history[previous_index] + 1
        };
        if self.state.start_pause_frame >= 0
            && frame - self.state.start_pause_frame > self.config.force_segment
        {
            contribution = 0;
        }
        let cumulative = previous + contribution;
        let index = frame as usize % HISTORY_SIZE;
        self.state.cumulative_speech[index] = cumulative;
        self.state.silence_history[index] = silence_run;
        self.state.checked_frame_count = frame + 1;
        VadEvidenceFrame {
            frame,
            speech_probability,
            silence_probability,
            neural_speech,
            energy_active,
            speech_evidence,
            cumulative_speech: cumulative,
            silence_run,
        }
    }

    fn cumulative(&self, frame: i32) -> i32 {
        self.state.cumulative_speech[frame.max(0) as usize % HISTORY_SIZE]
    }

    fn silence_run(&self, frame: i32) -> i32 {
        self.state.silence_history[frame.max(0) as usize % HISTORY_SIZE]
    }

    fn speech_count(&self, frame: i32, window: i32) -> i32 {
        self.cumulative(frame) - self.cumulative((frame - window).max(0))
    }

    fn start_segment(&mut self, frame: i32) -> VadStatus {
        let lookback = frame - START_WINDOW;
        self.state.read_delay = if lookback > self.config.frame_start_margin {
            self.config.frame_start_margin + START_WINDOW + 1
        } else {
            frame + 1
        };
        self.state.segment_count += 1;
        let candidate_start = frame - self.state.read_delay;
        self.state.start_pause_frame = self
            .state
            .pre_end_pause_frame
            .max(candidate_start)
            + 1;
        self.state.end_pause_frame = -1;
        self.state.may_pause_frame = -1;
        self.state.real_pause_frame = frame - START_WINDOW.min(frame);
        self.state.pre_end_status = -1;
        self.state.delay_end_found = false;
        self.state.current_status = VadStatus::SpeechStart as i32;
        VadStatus::SpeechStart
    }

    fn pre_end_window(&self) -> i32 {
        let mut window = self.config.pre_speech_end;
        if self.config.speech_end <= window {
            window = self.config.speech_end - 1;
        }
        window.max(0)
    }

    fn check_pre_end(&mut self, frame: i32, window: i32) -> Option<VadStatus> {
        if self.config.pre_speech_end_on
            && window > self.config.end_gap
            && self.state.pre_end_status < 0
            && frame > window
            && self.cumulative(frame) == self.cumulative(frame - window)
        {
            self.state.pre_end_status = frame;
            Some(VadStatus::PreSpeechEnd)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OriginalVadEndpoint, VadEndpointConfig, VadStatus};

    #[test]
    fn packaged_ring_history_start_and_end_match_vendor() {
        let mut endpoint = OriginalVadEndpoint::new(VadEndpointConfig::default())
            .expect("endpoint");
        let energy = [vec![false; 200], vec![true; 600], vec![false; 400]].concat();
        let mut first_start = None;
        let mut first_end = None;
        for (frame, energy) in energy.into_iter().enumerate() {
            let evidence = endpoint.push_logits(0.0, 0.0, energy);
            let status = endpoint.step(evidence.frame, 1_200).expect("step");
            if status == VadStatus::SpeechStart && first_start.is_none() {
                first_start = Some((frame, endpoint.state().start_pause_frame));
            }
            if status == VadStatus::SpeechEnd && first_end.is_none() {
                first_end = Some((
                    frame,
                    endpoint.state().real_pause_frame,
                    endpoint.state().end_pause_frame,
                ));
            }
            endpoint.state_mut().current_frame = evidence.frame + 1;
            endpoint.state_mut().read_delay = 0;
        }
        assert_eq!(first_start, Some((214, 164)));
        assert_eq!(first_end, Some((843, 543, 573)));
    }

    #[test]
    fn flush_closes_active_segment() {
        let mut endpoint = OriginalVadEndpoint::new(VadEndpointConfig::default())
            .expect("endpoint");
        let mut statuses = Vec::new();
        for frame in 0..20 {
            let evidence = endpoint.push_logits(0.0, 0.0, true);
            statuses.push(endpoint.step(evidence.frame, 20).expect("step"));
            assert_eq!(frame, evidence.frame);
        }
        assert_eq!(statuses[15], VadStatus::SpeechStart);
        assert_eq!(statuses[19], VadStatus::SpeechEnd);
    }
}
