use anyhow::{Result, bail};

const START_CHUNK: usize = 0;
const START_FRAME: usize = 1;
const END_CHUNK: usize = 2;
const END_FRAME: usize = 3;
const SCAN_CHUNK: usize = 4;
const SCAN_FRAME: usize = 5;
const PREVIOUS_END_CHUNK: usize = 6;
const PREVIOUS_END_FRAME: usize = 7;
const FRAME_WIDTH: usize = 8;
const PHASE: usize = 9;

const TANH_NUMERATOR: [f32; 7] = [
    f32::from_bits(0xA59F25C0),
    f32::from_bits(0x2A61337E),
    f32::from_bits(0xAEBD37FF),
    f32::from_bits(0x335C0041),
    f32::from_bits(0x3779434A),
    f32::from_bits(0x3A270DED),
    f32::from_bits(0x3BA059DC),
];
const TANH_DENOMINATOR: [f32; 4] = [
    f32::from_bits(0x35A0D3D8),
    f32::from_bits(0x38F895D6),
    f32::from_bits(0x3B14AA05),
    f32::from_bits(0x3BA059DD),
];
const TANH_APPROXIMATION_MIN: f32 = f32::from_bits(0x38D1B717);

#[derive(Clone, Copy, Debug)]
pub struct MemoryAttentionConfig {
    pub max_beams: usize,
    pub sequence_limit: usize,
    pub hidden_size: usize,
    pub choose_threshold: f32,
    pub stop_threshold: f32,
    pub beta_threshold: f32,
    pub end_margin: usize,
}

impl Default for MemoryAttentionConfig {
    fn default() -> Self {
        Self {
            max_beams: 8,
            sequence_limit: 2_048,
            hidden_size: 512,
            choose_threshold: 0.1,
            stop_threshold: 0.9,
            beta_threshold: 0.05,
            end_margin: 16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryAttentionInput<'a> {
    pub conformer_output: &'a [f32],
    pub at_h: &'a [f32],
    pub chunk_at_h: &'a [f32],
    pub enc_mem_h: &'a [f32],
    pub conformer_mask: &'a [f32],
    pub chunk_count: usize,
    pub frame_width: usize,
    pub passthrough: &'a [f32],
    pub query: &'a [f32],
    pub query_stop: &'a [f32],
    pub lengths: &'a [i32],
    pub final_flush: bool,
}

#[derive(Clone, Debug)]
pub struct MemoryAttentionResult {
    pub ready: bool,
    pub max_end: isize,
    pub context: Vec<f32>,
    pub passthrough: Vec<f32>,
    pub flags: Vec<i32>,
    pub lengths: Vec<i32>,
}

impl MemoryAttentionResult {
    fn waiting() -> Self {
        Self {
            ready: false,
            max_end: -1,
            context: Vec::new(),
            passthrough: Vec::new(),
            flags: Vec::new(),
            lengths: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct MemoryAttentionState {
    status: Vec<[i32; 10]>,
    alpha_current: Vec<f32>,
    alpha_working: Vec<f32>,
    beta: Vec<f32>,
    choose_prefix: Vec<f32>,
    stop_product: Vec<f32>,
}

impl MemoryAttentionState {
    fn new(config: MemoryAttentionConfig) -> Self {
        let mut status = vec![[0; 10]; config.max_beams];
        for beam in &mut status {
            beam[PHASE] = 3;
        }
        Self {
            status,
            alpha_current: vec![0.0; config.max_beams * config.sequence_limit],
            alpha_working: vec![1.0; config.max_beams * config.sequence_limit],
            beta: vec![0.0; config.max_beams * config.sequence_limit],
            choose_prefix: vec![0.0; config.max_beams * config.hidden_size],
            stop_product: vec![0.0; config.max_beams],
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryTryAttention {
    config: MemoryAttentionConfig,
    weight_at_v: Vec<f32>,
    weight_at_v_stop: Vec<f32>,
    state: MemoryAttentionState,
}

impl MemoryTryAttention {
    pub fn new(
        weight_at_v: Vec<f32>,
        weight_at_v_stop: Vec<f32>,
        config: MemoryAttentionConfig,
    ) -> Result<Self> {
        if config.max_beams == 0
            || config.sequence_limit == 0
            || config.hidden_size == 0
            || config.end_margin == 0
        {
            bail!("MemoryTryAttention dimensions must be positive")
        }
        if weight_at_v.len() != config.hidden_size
            || weight_at_v_stop.len() != config.hidden_size
        {
            bail!("MemoryTryAttention weight size does not match hidden size")
        }
        Ok(Self {
            config,
            weight_at_v,
            weight_at_v_stop,
            state: MemoryAttentionState::new(config),
        })
    }

    pub fn config(&self) -> MemoryAttentionConfig {
        self.config
    }

    pub fn reset(&mut self) {
        self.state = MemoryAttentionState::new(self.config);
    }

    pub fn gather_beams(&mut self, parents: &[usize]) -> Result<()> {
        if parents.len() > self.config.max_beams
            || parents.iter().any(|parent| *parent >= self.config.max_beams)
        {
            bail!("MemoryTryAttention parent index is outside beam state")
        }
        for (target, parent) in parents.iter().copied().enumerate() {
            let source = self.state.status[parent];
            let old_end_chunk = self.state.status[target][END_CHUNK];
            let old_end_frame = self.state.status[target][END_FRAME];
            self.state.status[target][PREVIOUS_END_CHUNK] = old_end_chunk;
            self.state.status[target][PREVIOUS_END_FRAME] = old_end_frame;
            self.state.status[target][START_CHUNK..=SCAN_FRAME]
                .copy_from_slice(&source[START_CHUNK..=SCAN_FRAME]);
            self.state.status[target][PHASE] = source[PHASE];

            let mut frame_width = self.state.status[target][FRAME_WIDTH];
            let copy_end = if frame_width < 1 {
                frame_width = source[FRAME_WIDTH];
                (source[END_CHUNK] * frame_width + source[END_FRAME]).max(
                    source[PREVIOUS_END_CHUNK] * frame_width
                        + source[PREVIOUS_END_FRAME],
                )
            } else {
                (source[END_CHUNK] * frame_width + source[END_FRAME])
                    .max(old_end_chunk * frame_width + old_end_frame)
            };
            let copy_start = source[START_CHUNK] * frame_width + source[START_FRAME];
            let copy_start = usize::try_from(copy_start)
                .map_err(|_| anyhow::anyhow!("MemoryTryAttention copy range is negative"))?;
            let copy_end = usize::try_from(copy_end)
                .map_err(|_| anyhow::anyhow!("MemoryTryAttention copy range is negative"))?;
            self.validate_range(copy_start, copy_end)?;
            if copy_start < copy_end {
                let source_start = self.alpha_offset(parent, copy_start);
                let target_start = self.alpha_offset(target, copy_start);
                let values = self.state.alpha_current
                    [source_start..source_start + copy_end - copy_start]
                    .to_vec();
                self.state.alpha_working
                    [target_start..target_start + copy_end - copy_start]
                    .copy_from_slice(&values);
            }
        }
        Ok(())
    }

    pub fn step(&mut self, input: MemoryAttentionInput<'_>) -> Result<MemoryAttentionResult> {
        let frame_count = input
            .chunk_count
            .checked_mul(input.frame_width)
            .ok_or_else(|| anyhow::anyhow!("MemoryTryAttention frame count overflow"))?;
        if input.chunk_count == 0 || input.frame_width == 0 {
            bail!("MemoryTryAttention encoder bank must not be empty")
        }
        if frame_count > self.config.sequence_limit {
            bail!("MemoryTryAttention encoder bank exceeds sequence limit")
        }
        let encoder_size = frame_count * self.config.hidden_size;
        for (name, value) in [
            ("conformer_output", input.conformer_output),
            ("at_h", input.at_h),
            ("chunk_at_h", input.chunk_at_h),
            ("enc_mem_h", input.enc_mem_h),
        ] {
            if value.len() != encoder_size {
                bail!("{} does not match the encoder bank shape", name)
            }
        }
        if input.conformer_mask.len() != frame_count {
            bail!("conformer_mask does not match the encoder bank shape")
        }
        let active = input.lengths.len();
        if active == 0 || active > self.config.max_beams {
            bail!("MemoryTryAttention active beam count is invalid")
        }
        let beam_size = active * self.config.hidden_size;
        for (name, value) in [
            ("passthrough", input.passthrough),
            ("query", input.query),
            ("query_stop", input.query_stop),
        ] {
            if value.len() != beam_size {
                bail!("{} does not match the active beam shape", name)
            }
        }

        let mut all_ready = true;
        for beam in 0..active {
            let mut status = self.state.status[beam];
            if status[FRAME_WIDTH] < 1 {
                status[FRAME_WIDTH] = input.frame_width as i32;
                let start = self.alpha_offset(beam, 0);
                let end = start + self.config.sequence_limit;
                self.state.alpha_current[start..end]
                    .copy_from_slice(&self.state.alpha_working[start..end]);
            } else if status[FRAME_WIDTH] != input.frame_width as i32 {
                bail!("MemoryTryAttention frame width changed")
            }

            if status[PHASE] == 1 {
                self.state.status[beam] = status;
                continue;
            }

            if status[PHASE] == 3 {
                let mut scan = self.position(
                    status[START_CHUNK],
                    status[START_FRAME],
                    input.frame_width,
                )?;
                while scan < frame_count
                    && self.state.alpha_working[self.alpha_offset(beam, scan)]
                        < self.config.choose_threshold
                {
                    scan += 1;
                }
                if scan == frame_count {
                    status[START_CHUNK] = input.chunk_count as i32;
                    status[START_FRAME] = 0;
                    status[END_CHUNK] = input.chunk_count as i32;
                    status[END_FRAME] = 0;
                    status[SCAN_CHUNK] = input.chunk_count as i32;
                    status[SCAN_FRAME] = 0;
                    status[PHASE] = if input.final_flush { 1 } else { -1 };
                    all_ready &= input.final_flush;
                    self.state.status[beam] = status;
                    continue;
                }

                let start_chunk = scan / input.frame_width;
                let start_frame = scan % input.frame_width;
                let copy_end = self
                    .position(status[END_CHUNK], status[END_FRAME], input.frame_width)?
                    .max(self.position(
                        status[PREVIOUS_END_CHUNK],
                        status[PREVIOUS_END_FRAME],
                        input.frame_width,
                    )?);
                self.validate_range(scan, copy_end)?;
                if scan < copy_end {
                    let source = self.alpha_offset(beam, scan);
                    let values = self.state.alpha_working[source..source + copy_end - scan]
                        .to_vec();
                    self.state.alpha_current[source..source + copy_end - scan]
                        .copy_from_slice(&values);
                }
                let prefix = self.hidden_offset(beam);
                self.state.choose_prefix[prefix..prefix + self.config.hidden_size].fill(0.0);
                self.state.stop_product[beam] = 1.0;
                status[START_CHUNK] = start_chunk as i32;
                status[START_FRAME] = start_frame as i32;
                status[SCAN_CHUNK] = start_chunk as i32;
                status[SCAN_FRAME] = start_frame as i32;
            }

            let mut chunk = usize::try_from(status[SCAN_CHUNK])
                .map_err(|_| anyhow::anyhow!("MemoryTryAttention scan chunk is negative"))?;
            let mut frame = usize::try_from(status[SCAN_FRAME])
                .map_err(|_| anyhow::anyhow!("MemoryTryAttention scan frame is negative"))?;
            let mut stopped = false;
            while chunk < input.chunk_count {
                let begin = chunk * input.frame_width + frame;
                let end = (chunk + 1) * input.frame_width;
                let mut choose = Vec::with_capacity(end - begin);
                for position in begin..end {
                    let hidden = self.encoder_row(input.at_h, position);
                    let query = self.beam_row(input.query, beam);
                    let mut transformed = vec![0.0; self.config.hidden_size];
                    for channel in 0..self.config.hidden_size {
                        transformed[channel] = tanh_float32(hidden[channel] + query[channel]);
                    }
                    let mut logit = dot_float32(&transformed, &self.weight_at_v);
                    logit += 1.0e10 * (input.conformer_mask[position] - 1.0);
                    let alpha = self.state.alpha_current[self.alpha_offset(beam, position)];
                    choose.push(alpha * sigmoid(logit));
                }

                let prefix_start = self.hidden_offset(beam);
                let mut running = self.state.choose_prefix
                    [prefix_start..prefix_start + self.config.hidden_size]
                    .to_vec();
                let mut prefixes = Vec::with_capacity((end - begin) * self.config.hidden_size);
                for (local, position) in (begin..end).enumerate() {
                    prefixes.extend_from_slice(&running);
                    let memory = self.encoder_row(input.enc_mem_h, position);
                    for channel in 0..self.config.hidden_size {
                        running[channel] = fma_float32(memory[channel], choose[local], running[channel]);
                    }
                }
                self.state.choose_prefix
                    [prefix_start..prefix_start + self.config.hidden_size]
                    .copy_from_slice(&running);

                let mut product = self.state.stop_product[beam];
                let mut cumulative_stop = Vec::with_capacity(end - begin);
                for (local, position) in (begin..end).enumerate() {
                    let prefix = &prefixes[local * self.config.hidden_size
                        ..(local + 1) * self.config.hidden_size];
                    let query_stop = self.beam_row(input.query_stop, beam);
                    let chunk_hidden = self.encoder_row(input.chunk_at_h, position);
                    let mut transformed = vec![0.0; self.config.hidden_size];
                    for channel in 0..self.config.hidden_size {
                        transformed[channel] = tanh_float32(
                            prefix[channel] + query_stop[channel] + chunk_hidden[channel],
                        );
                    }
                    let alpha = self.state.alpha_current[self.alpha_offset(beam, position)];
                    let stop = alpha * sigmoid(dot_float32(
                        &transformed,
                        &self.weight_at_v_stop,
                    ));
                    product *= 1.0 - stop;
                    cumulative_stop.push(1.0 - product);
                }
                self.state.stop_product[beam] = product;

                for (local, position) in (begin..end).enumerate() {
                    let offset = self.alpha_offset(beam, position);
                    let new_alpha = cumulative_stop[local] * self.state.alpha_current[offset];
                    if new_alpha > self.config.stop_threshold {
                        frame = position % input.frame_width;
                        status[END_CHUNK] = chunk as i32;
                        status[END_FRAME] = frame as i32;
                        status[SCAN_CHUNK] = chunk as i32;
                        status[SCAN_FRAME] = frame as i32;
                        status[PHASE] = 1;
                        stopped = true;
                        break;
                    }
                    self.state.alpha_current[offset] = new_alpha;
                    self.state.beta[offset] = (1.0 - new_alpha) * choose[local];
                }
                if stopped {
                    break;
                }
                chunk += 1;
                frame = 0;
            }

            if !stopped {
                status[END_CHUNK] = input.chunk_count as i32;
                status[END_FRAME] = 0;
                status[SCAN_CHUNK] = input.chunk_count as i32;
                status[SCAN_FRAME] = 0;
                status[PHASE] = if input.final_flush { 1 } else { -1 };
                all_ready &= input.final_flush;
            }
            self.state.status[beam] = status;
        }

        if !all_ready {
            return Ok(MemoryAttentionResult::waiting());
        }

        let mut context = vec![0.0; beam_size];
        let mut flags = vec![0; active];
        let mut max_end = -1_isize;
        for (beam, flag) in flags.iter_mut().enumerate() {
            self.state.status[beam][PHASE] = 3;
            let status = self.state.status[beam];
            let begin = self.position(
                status[START_CHUNK],
                status[START_FRAME],
                input.frame_width,
            )?;
            let end = self.position(
                status[END_CHUNK],
                status[END_FRAME],
                input.frame_width,
            )?;
            self.validate_range(begin, end)?;
            for position in begin..end {
                let beta = self.state.beta[self.alpha_offset(beam, position)];
                let output = self.encoder_row(input.conformer_output, position);
                let destination = beam * self.config.hidden_size;
                for channel in 0..self.config.hidden_size {
                    context[destination + channel] = fma_float32(
                        beta,
                        output[channel],
                        context[destination + channel],
                    );
                }
            }
            let mut beta_sum = 0.0_f32;
            for position in begin..end {
                beta_sum += self.state.beta[self.alpha_offset(beam, position)];
            }
            if beta_sum < self.config.beta_threshold {
                *flag |= 1;
            }
            if input.final_flush && frame_count.saturating_sub(end) < self.config.end_margin {
                *flag |= 2;
            }
            max_end = max_end.max(end as isize);
        }
        Ok(MemoryAttentionResult {
            ready: true,
            max_end,
            context,
            passthrough: input.passthrough.to_vec(),
            flags,
            lengths: input.lengths.to_vec(),
        })
    }

    fn alpha_offset(&self, beam: usize, position: usize) -> usize {
        beam * self.config.sequence_limit + position
    }

    fn hidden_offset(&self, beam: usize) -> usize {
        beam * self.config.hidden_size
    }

    fn encoder_row<'a>(&self, values: &'a [f32], position: usize) -> &'a [f32] {
        let start = position * self.config.hidden_size;
        &values[start..start + self.config.hidden_size]
    }

    fn beam_row<'a>(&self, values: &'a [f32], beam: usize) -> &'a [f32] {
        let start = beam * self.config.hidden_size;
        &values[start..start + self.config.hidden_size]
    }

    fn position(&self, chunk: i32, frame: i32, frame_width: usize) -> Result<usize> {
        let chunk = usize::try_from(chunk)
            .map_err(|_| anyhow::anyhow!("MemoryTryAttention chunk is negative"))?;
        let frame = usize::try_from(frame)
            .map_err(|_| anyhow::anyhow!("MemoryTryAttention frame is negative"))?;
        Ok(chunk * frame_width + frame)
    }

    fn validate_range(&self, begin: usize, end: usize) -> Result<()> {
        if end < begin || end > self.config.sequence_limit {
            bail!("MemoryTryAttention state range is invalid")
        }
        Ok(())
    }
}

fn fma_float32(left: f32, right: f32, accumulator: f32) -> f32 {
    ((left as f64) * (right as f64) + accumulator as f64) as f32
}

fn tanh_float32(value: f32) -> f32 {
    let absolute = value.abs();
    let clamped = absolute.min(9.0);
    let squared = clamped * clamped;
    let mut numerator = fma_float32(squared, TANH_NUMERATOR[0], TANH_NUMERATOR[1]);
    let mut denominator = fma_float32(
        squared,
        TANH_DENOMINATOR[0],
        TANH_DENOMINATOR[1],
    );
    for coefficient in &TANH_NUMERATOR[2..] {
        numerator = fma_float32(squared, numerator, *coefficient);
    }
    for coefficient in &TANH_DENOMINATOR[2..] {
        denominator = fma_float32(squared, denominator, *coefficient);
    }
    let approximation = numerator * clamped / denominator;
    let signed = f32::from_bits(
        (approximation.to_bits() & 0x7FFF_FFFF) | (value.to_bits() & 0x8000_0000),
    );
    if absolute >= TANH_APPROXIMATION_MIN {
        signed
    } else {
        value
    }
}

fn dot_float32(values: &[f32], weights: &[f32]) -> f32 {
    let mut accumulators = [0.0_f32; 4];
    for (channel, (value, weight)) in values.iter().zip(weights).enumerate() {
        let lane = channel & 3;
        accumulators[lane] = fma_float32(*value, *weight, accumulators[lane]);
    }
    accumulators[3] + (accumulators[2] + (accumulators[1] + accumulators[0]))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / ((-value).exp() + 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        END_CHUNK, FRAME_WIDTH, MemoryAttentionConfig, MemoryAttentionInput,
        MemoryTryAttention,
    };

    #[test]
    fn gathered_new_beam_initializes_alpha_on_next_step() {
        let config = MemoryAttentionConfig {
            max_beams: 2,
            sequence_limit: 4,
            hidden_size: 1,
            choose_threshold: 0.1,
            stop_threshold: 0.9,
            beta_threshold: 0.05,
            end_margin: 1,
        };
        let mut attention = MemoryTryAttention::new(vec![0.0], vec![0.0], config)
            .expect("attention");
        attention.state.status[0][FRAME_WIDTH] = 2;
        attention.state.status[0][END_CHUNK] = 1;
        attention.state.alpha_current[..2].copy_from_slice(&[0.25, 0.75]);

        attention.gather_beams(&[0, 0]).expect("gather");

        assert_eq!(attention.state.status[1][FRAME_WIDTH], 0);
        assert_eq!(&attention.state.alpha_working[4..6], &[0.25, 0.75]);
    }

    #[test]
    fn final_flush_returns_context_and_flags() {
        let config = MemoryAttentionConfig {
            max_beams: 1,
            sequence_limit: 4,
            hidden_size: 2,
            choose_threshold: 0.1,
            stop_threshold: 0.9,
            beta_threshold: 0.05,
            end_margin: 1,
        };
        let mut attention = MemoryTryAttention::new(vec![0.0; 2], vec![0.0; 2], config)
            .expect("attention");
        let result = attention
            .step(MemoryAttentionInput {
                conformer_output: &[1.0, 2.0, 3.0, 4.0],
                at_h: &[0.0; 4],
                chunk_at_h: &[0.0; 4],
                enc_mem_h: &[0.0; 4],
                conformer_mask: &[1.0, 1.0],
                chunk_count: 1,
                frame_width: 2,
                passthrough: &[0.0, 0.0],
                query: &[0.0, 0.0],
                query_stop: &[0.0, 0.0],
                lengths: &[0],
                final_flush: true,
            })
            .expect("step");
        assert!(result.ready);
        assert_eq!(result.context.len(), 2);
        assert_eq!(result.flags.len(), 1);
    }
}
