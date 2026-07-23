pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_LENGTH: usize = 400;
pub const FRAME_SHIFT: usize = 160;
pub const FFT_LENGTH: usize = 512;
pub const FEATURE_SIZE: usize = 80;
pub const VAD_FEATURE_SIZE: usize = 40;

const TABLE_DATA: &[u8; 5_272] = include_bytes!(
    "../data/original_frontend_tables.bin"
);
const LOG_OFFSET: usize = 0x0000;
const WINDOW_OFFSET: usize = 0x0800;
const BIT_REVERSE_OFFSET: usize = 0x0B20;
const WEIGHTS_40_OFFSET: usize = 0x0BA0;
const WEIGHTS_80_OFFSET: usize = 0x0D9E;
const MAP_40_OFFSET: usize = 0x0F9C;
const MAP_80_OFFSET: usize = 0x109B;
const TWIDDLE_COS_OFFSET: usize = 0x119A;
const TWIDDLE_SIN_OFFSET: usize = 0x129A;

#[derive(Clone, Debug)]
pub struct OriginalFeatureExtractor {
    feature_size: usize,
}

impl Default for OriginalFeatureExtractor {
    fn default() -> Self {
        Self {
            feature_size: FEATURE_SIZE,
        }
    }
}

impl OriginalFeatureExtractor {
    pub fn with_feature_size(feature_size: usize) -> anyhow::Result<Self> {
        if !matches!(feature_size, VAD_FEATURE_SIZE | FEATURE_SIZE) {
            anyhow::bail!("EdgeEsr frontend feature size must be 40 or 80")
        }
        Ok(Self { feature_size })
    }

    pub fn feature_size(&self) -> usize {
        self.feature_size
    }

    pub fn extract_frame_q22(&self, samples: &[i16]) -> anyhow::Result<Vec<i32>> {
        if samples.len() < FRAME_LENGTH {
            anyhow::bail!("EdgeEsr feature frame must contain at least 400 samples")
        }
        let (mut real, mut imaginary, mut exponent) = preprocess(samples);
        exponent = complex_fft(&mut real, &mut imaginary, exponent);
        real_fft(&mut real, &mut imaginary);
        Ok(filterbank_log(
            &real,
            &imaginary,
            exponent,
            self.feature_size,
        ))
    }

    pub fn extract_frame(&self, samples: &[i16]) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .extract_frame_q22(samples)?
            .into_iter()
            .map(|value| value.max(0) as f32 * 2.0_f32.powi(-22))
            .collect())
    }

    pub fn extract_frames(&self, samples: &[i16]) -> anyhow::Result<Vec<f32>> {
        if samples.len() < FRAME_LENGTH {
            return Ok(Vec::new());
        }
        let frame_count = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut output = Vec::with_capacity(frame_count * self.feature_size);
        for start in (0..=samples.len() - FRAME_LENGTH).step_by(FRAME_SHIFT) {
            output.extend(self.extract_frame(&samples[start..start + FRAME_LENGTH])?);
        }
        Ok(output)
    }
}

fn table_i16(offset: usize, index: usize) -> i16 {
    let start = offset + index * 2;
    i16::from_le_bytes([TABLE_DATA[start], TABLE_DATA[start + 1]])
}

fn window(index: usize) -> i16 {
    table_i16(WINDOW_OFFSET, index)
}

fn bit_reverse(index: usize) -> usize {
    TABLE_DATA[BIT_REVERSE_OFFSET + index] as usize
}

fn filter_weight(feature_size: usize, index: usize) -> i16 {
    let offset = if feature_size == VAD_FEATURE_SIZE {
        WEIGHTS_40_OFFSET
    } else {
        WEIGHTS_80_OFFSET
    };
    table_i16(offset, index)
}

fn filter_index(feature_size: usize, index: usize) -> usize {
    let offset = if feature_size == VAD_FEATURE_SIZE {
        MAP_40_OFFSET
    } else {
        MAP_80_OFFSET
    };
    TABLE_DATA[offset + index] as usize
}

fn twiddle_cos(index: usize) -> i16 {
    table_i16(TWIDDLE_COS_OFFSET, index)
}

fn twiddle_sin(index: usize) -> i16 {
    table_i16(TWIDDLE_SIN_OFFSET, index)
}

fn log_lut(index: usize) -> i16 {
    table_i16(LOG_OFFSET, index)
}

fn normalization_shift(bits: u32, previous_bits: u32) -> i32 {
    if bits == 0 {
        return 17;
    }
    if bits == u32::MAX {
        return -14;
    }
    let sign_mask = if (previous_bits as i32) < 0 {
        u32::MAX
    } else {
        0
    };
    let mut value = bits ^ sign_mask;
    if value as i32 >= 0x4000_0000 {
        return 17;
    }
    let mut count = 0_i32;
    while (value as i32) < 0x4000_0000 {
        value = value.wrapping_shl(1);
        count += 1;
    }
    17 - (count & 0xff)
}

fn preprocess(samples: &[i16]) -> ([i16; 257], [i16; 257], i32) {
    let mut work = [0_i32; FFT_LENGTH];
    for (target, sample) in work.iter_mut().zip(samples.iter().copied()) {
        *target = sample as i32;
    }
    let mean = work[..FRAME_LENGTH]
        .iter()
        .map(|value| *value as i64)
        .sum::<i64>()
        / FRAME_LENGTH as i64;
    let mean = mean as i32;

    let mut previous = work[FRAME_LENGTH - 1].wrapping_sub(mean);
    let mut bits = 0x8000_u32;
    let mut previous_bits = bits;
    let mut current = 0_i32;
    for index in (1..FRAME_LENGTH).rev() {
        current = work[index - 1].wrapping_sub(mean);
        let preemphasized = current
            .wrapping_mul(-0x7C29)
            .wrapping_add(previous.wrapping_mul(0x8000));
        let coefficient = window(index) as i32;
        let value = (((preemphasized as u32 & 0xffff) as i64 * coefficient as i64) >> 15)
            .wrapping_add((coefficient as i64) * ((preemphasized >> 16) as i64) * 2)
            as i32;
        work[index] = value;
        let absolute = if value < 0 {
            value.wrapping_neg()
        } else {
            value
        };
        previous_bits = bits;
        bits |= absolute as u32;
        previous = current;
    }
    work[0] = current.wrapping_mul(window(0) as i32);

    let scale = normalization_shift(bits, previous_bits);
    let rounding = 1_u32.wrapping_shl(((scale - 1) & 31) as u32) as i32;
    let shift = (scale & 31) as u32;
    let mut real = [0_i16; 257];
    let mut imaginary = [0_i16; 257];
    for index in 0..128 {
        let reversed_index = bit_reverse(index);
        let source_real = work[reversed_index].wrapping_add(rounding);
        let source_imaginary = work[reversed_index + 1].wrapping_add(rounding);
        real[index * 2] = source_real
            .wrapping_add(work[reversed_index + 256])
            .wrapping_shr(shift) as i16;
        real[index * 2 + 1] = source_real
            .wrapping_sub(work[reversed_index + 256])
            .wrapping_shr(shift) as i16;
        imaginary[index * 2] = source_imaginary
            .wrapping_add(work[reversed_index + 257])
            .wrapping_shr(shift) as i16;
        imaginary[index * 2 + 1] = source_imaginary
            .wrapping_sub(work[reversed_index + 257])
            .wrapping_shr(shift) as i16;
    }
    (real, imaginary, 15 - scale)
}

fn complex_fft(real: &mut [i16; 257], imaginary: &mut [i16; 257], mut exponent: i32) -> i32 {
    let mut size = 4_usize;
    let mut half_size = 2_usize;
    let mut twiddle_shift = 7_u32;
    while size <= 256 {
        for group in 0..half_size {
            let twiddle_index = ((group as i32).wrapping_shl(twiddle_shift) as i16) as usize;
            let cosine = twiddle_cos(twiddle_index) as i32;
            let sine = twiddle_sin(twiddle_index) as i32;
            for left in (group..256).step_by(size) {
                let right = left + half_size;
                let left_real = real[left] as i32;
                let left_imaginary = imaginary[left] as i32;
                let right_real = real[right] as i32;
                let right_imaginary = imaginary[right] as i32;
                let real_product = right_real
                    .wrapping_mul(cosine)
                    .wrapping_add(0x4000)
                    .wrapping_sub(right_imaginary.wrapping_mul(sine));
                let imaginary_product = right_imaginary
                    .wrapping_mul(cosine)
                    .wrapping_add(right_real.wrapping_mul(sine))
                    .wrapping_add(0x4000);
                if twiddle_shift == 4 {
                    let rotated_real = (real_product as u32 >> 15) as i16 as i32;
                    let rotated_imaginary = (imaginary_product as u32 >> 15) as i16 as i32;
                    real[right] = left_real.wrapping_sub(rotated_real) as i16;
                    imaginary[right] = left_imaginary.wrapping_sub(rotated_imaginary) as i16;
                    real[left] = left_real.wrapping_add(rotated_real) as i16;
                    imaginary[left] = left_imaginary.wrapping_add(rotated_imaginary) as i16;
                } else {
                    let rotated_real = real_product >> 15;
                    let rotated_imaginary = imaginary_product >> 15;
                    real[right] = (left_real
                        .wrapping_add(1)
                        .wrapping_sub(rotated_real) as u32
                        >> 1) as i16;
                    imaginary[right] = (left_imaginary
                        .wrapping_add(1)
                        .wrapping_sub(rotated_imaginary) as u32
                        >> 1) as i16;
                    real[left] = (rotated_real
                        .wrapping_add(left_real)
                        .wrapping_add(1) as u32
                        >> 1) as i16;
                    imaginary[left] = (rotated_imaginary
                        .wrapping_add(left_imaginary)
                        .wrapping_add(1) as u32
                        >> 1) as i16;
                }
            }
        }
        if twiddle_shift != 4 {
            exponent -= 1;
        }
        size *= 2;
        half_size *= 2;
        twiddle_shift -= 1;
    }
    exponent
}

fn real_fft(real: &mut [i16; 257], imaginary: &mut [i16; 257]) {
    real[256] = real[0];
    imaginary[256] = imaginary[0];
    for index in 0..128 {
        let mirror = 256 - index;
        let current_real = real[index] as i32;
        let mirror_real = real[mirror] as i32;
        let current_imaginary = imaginary[index] as i32;
        let mirror_imaginary = imaginary[mirror] as i32;
        let cosine = twiddle_cos(index) as i32;
        let sine = twiddle_sin(index) as i32;

        let average_imaginary = current_imaginary
            .wrapping_add(1)
            .wrapping_add(mirror_imaginary)
            >> 1;
        let difference_real = 1_i32
            .wrapping_sub(current_real)
            .wrapping_add(mirror_real)
            >> 1;
        let average_real = current_real
            .wrapping_add(mirror_real)
            .wrapping_add(1)
            .wrapping_shr(1) as i16 as i32;
        let difference_imaginary = current_imaginary
            .wrapping_add(1)
            .wrapping_sub(mirror_imaginary)
            .wrapping_shr(1) as i16 as i32;

        let positive_rotation = 0x4000_i32
            .wrapping_sub(difference_real.wrapping_mul(sine))
            .wrapping_add(average_imaginary.wrapping_mul(cosine));
        let negative_rotation = difference_real
            .wrapping_mul(sine)
            .wrapping_add(0x4000)
            .wrapping_sub(average_imaginary.wrapping_mul(cosine));
        let imaginary_rotation = average_imaginary
            .wrapping_mul(sine)
            .wrapping_add(difference_real.wrapping_mul(cosine))
            .wrapping_add(0x4000)
            .wrapping_shr(15) as i16 as i32;

        real[index] = average_real
            .wrapping_add((positive_rotation as u32 >> 15) as i32)
            as i16;
        imaginary[index] = imaginary_rotation.wrapping_add(difference_imaginary) as i16;
        real[mirror] = average_real
            .wrapping_add((negative_rotation as u32 >> 15) as i32)
            as i16;
        imaginary[mirror] = imaginary_rotation.wrapping_sub(difference_imaginary) as i16;
    }
}

fn filterbank_log(
    real: &[i16; 257],
    imaginary: &[i16; 257],
    exponent: i32,
    feature_size: usize,
) -> Vec<i32> {
    let mut accumulators = vec![0_i32; feature_size];
    for index in 0..255 {
        let real_value = real[index + 1] as i32;
        let imaginary_value = imaginary[index + 1] as i32;
        let power = real_value
            .wrapping_mul(real_value)
            .wrapping_add(imaginary_value.wrapping_mul(imaginary_value)) as u32;
        let weight = filter_weight(feature_size, index) as i32;
        let weighted = ((((power & 0xffff) as i64 * weight as i64) >> 15)
            + weight as i64 * (power >> 16) as i64 * 2) as i32;
        let filter = filter_index(feature_size, index);
        if filter < feature_size {
            accumulators[filter] = (accumulators[filter] as i64
                + power as i64
                - weighted as i64) as i32;
        }
        if filter > 0 && filter <= feature_size {
            accumulators[filter - 1] = accumulators[filter - 1].wrapping_add(weighted);
        }
    }

    let base_shift = exponent.wrapping_mul(2) as i8;
    accumulators
        .into_iter()
        .map(|accumulator| {
            let mut value = accumulator.wrapping_add(2) as u32;
            let mut shift = base_shift as i16;
            if value >> 16 == 0 {
                value = value.wrapping_shl(16);
                shift += 16;
            }
            if value >> 24 == 0 {
                value = value.wrapping_shl(8);
                shift += 8;
            }
            if value >> 28 == 0 {
                value = value.wrapping_shl(4);
                shift += 4;
            }
            if value >> 30 == 0 {
                value = value.wrapping_shl(2);
                shift += 2;
            }
            let final_shift = ((value >> 31) ^ 1) as i16;
            value = value.wrapping_shl(final_shift as u32);
            shift += final_shift;
            let lookup_index = ((value >> 21) ^ 0x400) as usize;
            (31_i32.wrapping_sub(shift as i32))
                .wrapping_mul(0x2C5C86)
                .wrapping_add((log_lut(lookup_index) as i32).wrapping_mul(0x80))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_FIXTURE: &[u8; 2_048] = include_bytes!(
        "../test-data/original_frontend_frame_i32.bin"
    );
    const FEATURE_Q22_FIXTURE: &[u8; 320] = include_bytes!(
        "../test-data/original_frontend_feature_q22.bin"
    );
    const VAD_Q22_FIXTURE: &[u8; 160] = include_bytes!(
        "../test-data/original_vad_frontend_feature_q22.bin"
    );

    fn fixture_frame() -> Vec<i16> {
        FRAME_FIXTURE
            .chunks_exact(4)
            .take(FRAME_LENGTH)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i16)
            .collect()
    }

    fn fixture_i32(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn frontend_matches_original_q22_fixture() {
        let actual = OriginalFeatureExtractor::default()
            .extract_frame_q22(&fixture_frame())
            .unwrap();
        assert_eq!(actual, fixture_i32(FEATURE_Q22_FIXTURE));
    }

    #[test]
    fn vad_frontend_matches_original_q22_fixture() {
        let actual = OriginalFeatureExtractor::with_feature_size(VAD_FEATURE_SIZE)
            .unwrap()
            .extract_frame_q22(&fixture_frame())
            .unwrap();
        assert_eq!(actual, fixture_i32(VAD_Q22_FIXTURE));
    }
}
