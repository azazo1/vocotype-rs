use std::cell::RefCell;

use half::f16;

#[derive(Default)]
struct PackedGemmWorkspace {
    packed_left: Vec<f32>,
    right: Vec<f32>,
    bias: Vec<f32>,
}

thread_local! {
    static PACKED_GEMM_WORKSPACE: RefCell<PackedGemmWorkspace> = RefCell::new(PackedGemmWorkspace::default());
}

pub(crate) const fn available() -> bool {
    cfg!(target_arch = "aarch64")
}

pub(crate) fn convert_half_to_float(input: &[f16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_half_to_float_neon(input, output) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (target, value) in output.iter_mut().zip(input) {
        *target = value.to_f32();
    }
}

pub(crate) fn convert_float_to_half(input: &[f32], output: &mut [f16]) {
    assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_float_to_half_neon(input, output) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (target, value) in output.iter_mut().zip(input) {
        *target = f16::from_f32(*value);
    }
}

pub(crate) fn packed_gemm_f16_into(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
    output: &mut [f16],
) -> bool {
    if !available() || rows < 8 || !rows.is_multiple_of(8) {
        return false;
    }
    PACKED_GEMM_WORKSPACE.with_borrow_mut(|workspace| {
        workspace.packed_left.resize(rows * inner, 0.0);
        workspace.right.resize(columns * inner, 0.0);
        workspace.bias.resize(columns, 0.0);
        for row in 0..rows {
            for index in 0..inner {
                workspace.packed_left[index * rows + row] = left[row * inner + index].to_f32();
            }
        }
        convert_half_to_float(right, &mut workspace.right);
        convert_half_to_float(bias, &mut workspace.bias);
        for column in 0..columns {
            for row in (0..rows).step_by(8) {
                let values = gemm_row_block(
                    &workspace.packed_left,
                    &workspace.right,
                    workspace.bias[column],
                    rows,
                    inner,
                    column,
                    row,
                );
                for lane in 0..8 {
                    output[(row + lane) * columns + column] = f16::from_f32(values[lane]);
                }
            }
        }
    });
    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_half_to_float_neon(input: &[f16], output: &mut [f32]) {
    use core::arch::aarch64::{
        vcvt_f32_f16, vcvt_high_f32_f16, vget_low_f16, vld1q_u16,
        vreinterpretq_f16_u16, vst1q_f32,
    };

    let mut index = 0;
    while index + 8 <= input.len() {
        let values = unsafe { vld1q_u16(input.as_ptr().add(index).cast::<u16>()) };
        let values = vreinterpretq_f16_u16(values);
        unsafe {
            vst1q_f32(output.as_mut_ptr().add(index), vcvt_f32_f16(vget_low_f16(values)));
            vst1q_f32(output.as_mut_ptr().add(index + 4), vcvt_high_f32_f16(values));
        }
        index += 8;
    }
    for (target, value) in output[index..].iter_mut().zip(&input[index..]) {
        *target = value.to_f32();
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_float_to_half_neon(input: &[f32], output: &mut [f16]) {
    use core::arch::aarch64::{
        vcvt_f16_f32, vcvt_high_f16_f32, vld1q_f32, vreinterpretq_u16_f16, vst1q_u16,
    };

    let mut index = 0;
    while index + 8 <= input.len() {
        let low = unsafe { vld1q_f32(input.as_ptr().add(index)) };
        let high = unsafe { vld1q_f32(input.as_ptr().add(index + 4)) };
        let values = vcvt_high_f16_f32(vcvt_f16_f32(low), high);
        unsafe {
            vst1q_u16(
                output.as_mut_ptr().add(index).cast::<u16>(),
                vreinterpretq_u16_f16(values),
            );
        }
        index += 8;
    }
    for (target, value) in output[index..].iter_mut().zip(&input[index..]) {
        *target = f16::from_f32(*value);
    }
}

fn gemm_row_block(
    packed_left: &[f32],
    right: &[f32],
    bias: f32,
    rows: usize,
    inner: usize,
    column: usize,
    row: usize,
) -> [f32; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            gemm_row_block_neon(
                packed_left,
                right,
                bias,
                rows,
                inner,
                column,
                row,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut output = [bias; 8];
        let right = &right[column * inner..(column + 1) * inner];
        for index in 0..inner {
            for lane in 0..8 {
                output[lane] = packed_left[index * rows + row + lane]
                    .mul_add(right[index], output[lane]);
            }
        }
        output
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_row_block_neon(
    packed_left: &[f32],
    right: &[f32],
    bias: f32,
    rows: usize,
    inner: usize,
    column: usize,
    row: usize,
) -> [f32; 8] {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_n_f32, vld1q_f32, vst1q_f32};

    let mut low = vdupq_n_f32(bias);
    let mut high = low;
    let right = &right[column * inner..(column + 1) * inner];
    for (index, weight) in right.iter().copied().enumerate().take(inner) {
        let left = unsafe { packed_left.as_ptr().add(index * rows + row) };
        low = vfmaq_n_f32(low, unsafe { vld1q_f32(left) }, weight);
        high = vfmaq_n_f32(high, unsafe { vld1q_f32(left.add(4)) }, weight);
    }
    let mut output = [0.0; 8];
    unsafe {
        vst1q_f32(output.as_mut_ptr(), low);
        vst1q_f32(output.as_mut_ptr().add(4), high);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_float_conversion_round_trips() {
        let input = (0..19)
            .map(|index| f16::from_f32(index as f32 * 0.125 - 1.0))
            .collect::<Vec<_>>();
        let mut floats = vec![0.0; input.len()];
        let mut output = vec![f16::ZERO; input.len()];
        convert_half_to_float(&input, &mut floats);
        convert_float_to_half(&floats, &mut output);
        assert_eq!(output, input);
    }
}
