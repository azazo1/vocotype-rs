use half::f16;

const GELU_LUT: &[u8; 131_072] = include_bytes!(
    "../data/xlite_gelu_fp16_lut.bin"
);
const EXP2_PERIODIC_CORRECTIONS: &[u8; 2_097_152] = include_bytes!(
    "../data/xlite_exp2_f32_periodic_corrections.bin"
);
const EXP2_SUBUNIT_CORRECTIONS: &[u8; 2_907_270] = include_bytes!(
    "../data/xlite_exp2_f32_subunit_corrections.bin"
);
const RECIPROCAL_CORRECTIONS: &[u8; 2_097_152] = include_bytes!(
    "../data/xlite_reciprocal_f32_corrections.bin"
);
const RSQRT_CORRECTIONS: &[u8; 4_194_304] = include_bytes!(
    "../data/xlite_rsqrt_f32_corrections.bin"
);
const EXP2_SUBUNIT_ENTRIES: u32 = 11_629_080;

pub(crate) fn gelu(value: f16) -> f16 {
    let offset = value.to_bits() as usize * 2;
    f16::from_bits(u16::from_le_bytes([GELU_LUT[offset], GELU_LUT[offset + 1]]))
}

pub(crate) fn original_sigmoid(value: f16) -> f16 {
    let negative = f16::from_bits(value.to_bits() ^ 0x8000);
    let exponential = f16::from_f32(negative.to_f32().exp());
    let denominator = f16::from_f32(1.0 + exponential.to_f32());
    f16::from_f64(1.0 / denominator.to_f32() as f64)
}

pub(crate) fn decoder_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + bionic_exponential(-value))
}

pub(crate) fn softmax_exponential(value: f32) -> f32 {
    let log2e = f32::from_bits(0x3FB8AA3B);
    let scaled = value * log2e;
    let exponential2 = softmax_exponential2(value, scaled);
    let reduction_error = value.mul_add(
        f32::from_bits(0x32A57060),
        value.mul_add(log2e, -scaled),
    );
    let correction = reduction_error * f32::from_bits(0x3F317218);
    let result = exponential2.mul_add(correction, exponential2);
    if result.is_nan() { exponential2 } else { result }
}

pub(crate) fn softmax_reciprocal(value: f32) -> f32 {
    let result = (1.0_f64 / value as f64) as f32;
    if value <= 0.0 || !value.is_finite() || value.is_subnormal() {
        return result;
    }
    apply_packed_correction(result, value.to_bits() & 0x7F_FFFF, RECIPROCAL_CORRECTIONS)
}

pub(crate) fn layer_norm_rsqrt(value: f32) -> f32 {
    let result = (1.0_f64 / (value as f64).sqrt()) as f32;
    if value <= 0.0 || !value.is_finite() || value.is_subnormal() {
        return result;
    }
    let bits = value.to_bits();
    let index = (((bits >> 23) & 1) << 23) | (bits & 0x7F_FFFF);
    apply_packed_correction(result, index, RSQRT_CORRECTIONS)
}

pub(crate) fn decoder_log_softmax_denominator(values: &[f32], maximum: f32) -> f32 {
    let clamp_min = f32::from_bits(0xC2B0C0A5);
    let clamp_max = f32::from_bits(0x42B0C0A5);
    let log2e = f32::from_bits(0x3FB8AA3B);
    let ln2_high = f32::from_bits(0x3F318000);
    let ln2_low = f32::from_bits(0xB95E8083);
    let coefficients = [
        f32::from_bits(0x39506967),
        f32::from_bits(0x3AB743CE),
        f32::from_bits(0x3C088908),
        f32::from_bits(0x3D2AA9C1),
        f32::from_bits(0x3E2AAAAA),
    ];
    let mut lane_sums = [0.0_f32; 4];
    let vector_width = values.len() & !3;
    for index in (0..vector_width).step_by(4) {
        for lane in 0..4 {
            let shifted = values[index + lane] - maximum;
            let reduced = shifted.clamp(clamp_min, clamp_max);
            let scaled = reduced * log2e;
            let rounded = scaled + 0.5;
            let mut exponent_float = (rounded as i32) as f32;
            exponent_float -= f32::from(exponent_float > rounded);
            let exponent = exponent_float as i32;
            let mut remainder = reduced - exponent_float * ln2_high;
            remainder -= exponent_float * ln2_low;
            let mut polynomial = remainder * coefficients[0];
            for coefficient in &coefficients[1..] {
                polynomial += *coefficient;
                polynomial *= remainder;
            }
            polynomial += 0.5;
            let squared = remainder * remainder;
            let mut approximation = remainder + squared * polynomial;
            approximation += 1.0;
            let power_bits = (i64::from(0x3F80_0000_u32)
                + i64::from(exponent) * 0x80_0000) as u32;
            approximation *= f32::from_bits(power_bits);
            lane_sums[lane] += approximation;
        }
    }
    let mut denominator = (lane_sums[0] + lane_sums[1]) + (lane_sums[2] + lane_sums[3]);
    for value in &values[vector_width..] {
        denominator += (*value - maximum).exp();
    }
    denominator
}

fn softmax_exponential2(value: f32, scaled: f32) -> f32 {
    if scaled.is_nan() || scaled > 0.0 {
        return scaled.exp2();
    }
    if scaled < -126.0 {
        return 0.0;
    }
    let result = scaled.exp2();
    if scaled > -1.0 {
        let index = (-value * 16_777_216.0) as u32;
        if index >= EXP2_SUBUNIT_ENTRIES {
            return result;
        }
        return apply_packed_correction(result, index, EXP2_SUBUNIT_CORRECTIONS);
    }
    let fraction = scaled - scaled.floor();
    let index = (fraction * 8_388_608.0) as u32;
    apply_packed_correction(result, index, EXP2_PERIODIC_CORRECTIONS)
}

fn apply_packed_correction(value: f32, index: u32, corrections: &[u8]) -> f32 {
    let byte_index = (index >> 2) as usize;
    let Some(byte) = corrections.get(byte_index) else {
        return value;
    };
    let code = (byte >> ((index & 3) * 2)) & 3;
    let bits = match code {
        0 => return value,
        1 => value.to_bits().wrapping_add(1),
        2 => value.to_bits().wrapping_sub(1),
        _ => value.to_bits().wrapping_sub(2),
    };
    f32::from_bits(bits)
}

fn bionic_exponential(value: f32) -> f32 {
    let one = f32::from_bits(0x3F80_0000);
    let huge = f32::from_bits(0x7149_F2CA);
    let two_minus_100 = f32::from_bits(0x0D80_0000);
    let mut x = value;
    let bits = x.to_bits();
    let sign = bits >> 31;
    let magnitude = bits & 0x7FFF_FFFF;
    if magnitude >= 0x42B1_7218 {
        if magnitude > 0x7F80_0000 {
            return x + x;
        }
        if magnitude == 0x7F80_0000 {
            return if sign == 0 { x } else { 0.0 };
        }
        if x > f32::from_bits(0x42B1_7180) {
            return huge * huge;
        }
        if x < f32::from_bits(0xC2CF_F1B5) {
            return two_minus_100 * two_minus_100;
        }
    }

    let mut high = 0.0_f32;
    let mut low = 0.0_f32;
    let mut exponent = 0_i32;
    if magnitude > 0x3EB1_7218 {
        let ln2_high = f32::from_bits(0x3F31_7200);
        let ln2_low = f32::from_bits(0x35BF_BE8E);
        if magnitude < 0x3F85_1592 {
            high = x - if sign == 0 { ln2_high } else { -ln2_high };
            low = if sign == 0 { ln2_low } else { -ln2_low };
            exponent = 1 - sign as i32 * 2;
        } else {
            let half = if sign == 0 { 0.5 } else { -0.5 };
            exponent = (f32::from_bits(0x3FB8_AA3B) * x + half) as i32;
            let exponent_float = exponent as f32;
            high = x - exponent_float * ln2_high;
            low = exponent_float * ln2_low;
        }
        x = high - low;
    } else if magnitude < 0x3900_0000 && huge + x > one {
        return one + x;
    }

    let squared = x * x;
    let correction = x
        - squared
            * (f32::from_bits(0x3E2A_AA8F)
                + squared * f32::from_bits(0xBB35_5215));
    let product = x * correction;
    if exponent == 0 {
        return one - (product / (correction - 2.0) - x);
    }
    let result = one - ((low - product / (2.0 - correction)) - high);
    if exponent >= -125 {
        if exponent == 128 {
            return (result * 2.0) * f32::from_bits(0x7F00_0000);
        }
        let scale_bits = (i64::from(0x3F80_0000_u32)
            + i64::from(exponent) * 0x80_0000) as u32;
        return result * f32::from_bits(scale_bits);
    }
    let scale_bits = (i64::from(0x3F80_0000_u32)
        + i64::from(exponent + 100) * 0x80_0000) as u32;
    (result * f32::from_bits(scale_bits)) * two_minus_100
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{gelu, layer_norm_rsqrt, softmax_exponential, softmax_reciprocal};

    #[test]
    fn gelu_table_covers_all_half_bit_patterns() {
        for bits in 0_u16..=u16::MAX {
            let _ = gelu(f16::from_bits(bits));
        }
    }

    #[test]
    fn corrected_primitives_keep_basic_identities() {
        assert_eq!(softmax_exponential(0.0), 1.0);
        assert!((softmax_reciprocal(2.0) - 0.5).abs() <= f32::EPSILON);
        assert!((layer_norm_rsqrt(4.0) - 0.5).abs() <= f32::EPSILON);
    }
}
