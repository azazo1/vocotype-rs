use std::sync::atomic::{AtomicI32, Ordering};

use half::f16;

use crate::numerics;

static DECODER_ACTIVE_ROWS: AtomicI32 = AtomicI32::new(1);

pub fn decoder_active_rows() -> i32 {
    DECODER_ACTIVE_ROWS.load(Ordering::Relaxed)
}

pub fn set_decoder_active_rows(rows: i32) -> anyhow::Result<()> {
    if rows <= 0 {
        anyhow::bail!("decoder active rows must be positive")
    }
    DECODER_ACTIVE_ROWS.store(rows, Ordering::Relaxed);
    Ok(())
}

fn rounded_add(left: f16, right: f16) -> f16 {
    f16::from_f32(left.to_f32() + right.to_f32())
}

fn rounded_mul(left: f16, right: f16) -> f16 {
    f16::from_f32(left.to_f32() * right.to_f32())
}

pub fn gemm_f16(left: &[f16], right: &[f16], bias: &[f16], rows: usize, inner: usize, columns: usize) -> anyhow::Result<Vec<f16>> {
    if left.len() != rows * inner || right.len() != columns * inner || bias.len() != columns {
        anyhow::bail!("invalid GEMM dimensions")
    }
    let mut output = vec![f16::from_f32(0.0); rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut value = bias[column].to_f32();
            for index in 0..inner {
                value = left[row * inner + index]
                    .to_f32()
                    .mul_add(right[column * inner + index].to_f32(), value);
            }
            output[row * columns + column] = f16::from_f32(value);
        }
    }
    Ok(output)
}

pub fn decoder_gemm(left: &[f16], right: &[f16], bias: &[f16], rows: usize, inner: usize, columns: usize) -> anyhow::Result<Vec<f16>> {
    if left.len() != rows * inner || right.len() != columns * inner || bias.len() != columns {
        anyhow::bail!("invalid decoder GEMM dimensions")
    }
    let active = decoder_active_rows() as usize;
    if active > rows {
        anyhow::bail!("decoder active rows exceed row count")
    }
    let paired = active == 1;
    if paired && !inner.is_multiple_of(4) {
        anyhow::bail!("one-row decoder GEMM requires an inner multiple of four")
    }
    let mut output = vec![f16::from_f32(0.0); rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut even = bias[column];
            let mut odd = f16::from_f32(0.0);
            for index in 0..inner {
                let accumulator = if paired && index & 1 != 0 {
                    &mut odd
                } else {
                    &mut even
                };
                *accumulator = f16::from_f32(
                    left[row * inner + index].to_f32().mul_add(
                        right[column * inner + index].to_f32(),
                        accumulator.to_f32(),
                    ),
                );
            }
            output[row * columns + column] = if paired {
                rounded_add(even, odd)
            } else {
                even
            };
        }
    }
    Ok(output)
}

pub fn matmul_f16(left: &[f16], right: &[f16], rows: usize, inner: usize, columns: usize) -> anyhow::Result<Vec<f16>> {
    if left.len() != rows * inner
        || right.len() != inner * columns
        || rows == 0
        || inner == 0
        || columns == 0
        || !inner.is_multiple_of(8)
    {
        anyhow::bail!("invalid MatMul dimensions")
    }
    let mut output = vec![f16::from_f32(0.0); rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut stash = 0.0_f32;
            for block in (0..inner).step_by(8) {
                let mut accumulator = f16::from_f32(0.0);
                for lane in 0..8 {
                    let index = block + lane;
                    accumulator = rounded_add(
                        rounded_mul(
                            left[row * inner + index],
                            right[index * columns + column],
                        ),
                        accumulator,
                    );
                }
                stash += accumulator.to_f32();
            }
            output[row * columns + column] = f16::from_f32(stash);
        }
    }
    Ok(output)
}

pub fn decoder_matmul(left: &[f32], right: &[f32], rows: usize, inner: usize, columns: usize) -> anyhow::Result<Vec<f32>> {
    if left.len() != rows * inner
        || right.len() != inner * columns
        || rows != 1
        || inner == 0
        || columns == 0
    {
        anyhow::bail!("invalid decoder MatMul dimensions")
    }
    let mut output = vec![0.0; rows * columns];
    for column in 0..columns {
        let mut result = 0.0_f32;
        if column < (columns & !1) {
            let mut lanes = [0.0_f32; 4];
            let main_inner = inner & !3;
            for index in (0..main_inner).step_by(4) {
                for lane in 0..4 {
                    lanes[lane] = left[index + lane].mul_add(
                        right[(index + lane) * columns + column],
                        lanes[lane],
                    );
                }
            }
            let pair_01 = lanes[0] + lanes[1];
            let pair_23 = lanes[2] + lanes[3];
            result = pair_01 + pair_23;
            for index in main_inner..inner {
                result = left[index].mul_add(
                    right[index * columns + column],
                    result,
                );
            }
        } else {
            for index in 0..inner {
                let product = left[index] * right[index * columns + column];
                result += product;
            }
        }
        output[column] = result;
    }
    Ok(output)
}

pub fn layer_norm_f16(input: &[f16], scale: &[f16], bias: &[f16], rows: usize, width: usize, epsilon: f32) -> anyhow::Result<Vec<f16>> {
    if input.len() != rows * width
        || scale.len() != width
        || bias.len() != width
        || !width.is_multiple_of(4)
    {
        anyhow::bail!("invalid layer normalization dimensions")
    }
    let mut output = vec![f16::from_f32(0.0); input.len()];
    for row in 0..rows {
        let offset = row * width;
        let mut mean_lanes = [0.0_f32; 4];
        let mut variance_lanes = [0.0_f32; 4];
        for index in (0..width).step_by(4) {
            for lane in 0..4 {
                let value = input[offset + index + lane].to_f32();
                mean_lanes[lane] += value;
                variance_lanes[lane] += value * value;
            }
        }
        let mut mean = mean_lanes[0];
        let mut second_moment = variance_lanes[0];
        for lane in 1..4 {
            mean += mean_lanes[lane];
            second_moment += variance_lanes[lane];
        }
        mean /= width as f32;
        let second_moment_mean = second_moment / width as f32;
        let variance = (-mean).mul_add(mean, second_moment_mean);
        let inverse = numerics::layer_norm_rsqrt(variance + epsilon);
        for index in 0..width {
            let centered = input[offset + index].to_f32() - mean;
            let scaled_inverse = scale[index].to_f32() * inverse;
            let value = centered * scaled_inverse + bias[index].to_f32();
            output[offset + index] = f16::from_f32(value);
        }
    }
    Ok(output)
}

pub fn decoder_layer_norm(input: &[f32], scale: &[f32], bias: &[f32], rows: usize, width: usize, epsilon: f32) -> anyhow::Result<Vec<f32>> {
    if input.len() != rows * width || scale.len() != width || bias.len() != width {
        anyhow::bail!("invalid decoder layer normalization dimensions")
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * width;
        let mean = input[offset..offset + width].iter().sum::<f32>() / width as f32;
        let variance = input[offset..offset + width]
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        let mean_inverse = mean * inverse;
        for index in 0..width {
            let normalized = input[offset + index] * inverse - mean_inverse;
            output[offset + index] = normalized * scale[index] + bias[index];
        }
    }
    Ok(output)
}

pub fn sigmoid_f16(input: &[f16]) -> Vec<f16> {
    input.iter().copied().map(numerics::original_sigmoid).collect()
}

pub fn decoder_sigmoid(input: &[f32]) -> Vec<f32> {
    input.iter().copied().map(numerics::decoder_sigmoid).collect()
}

pub fn decoder_cos(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| (*value as f64).cos() as f32).collect()
}

pub fn decoder_sin(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| (*value as f64).sin() as f32).collect()
}

pub fn softmax_f16(input: &[f16], rows: usize, width: usize) -> anyhow::Result<Vec<f16>> {
    if input.len() != rows * width || width == 0 {
        anyhow::bail!("invalid softmax dimensions")
    }
    let mut output = vec![f16::from_f32(0.0); input.len()];
    for row in 0..rows {
        let offset = row * width;
        let maximum = input[offset..offset + width]
            .iter()
            .map(|value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let values = input[offset..offset + width]
            .iter()
            .map(|value| numerics::softmax_exponential(value.to_f32() - maximum))
            .collect::<Vec<_>>();
        let denominator = values.iter().sum::<f32>();
        let reciprocal = numerics::softmax_reciprocal(denominator);
        for index in 0..width {
            output[offset + index] = f16::from_f32(values[index] * reciprocal);
        }
    }
    Ok(output)
}

pub fn decoder_log_softmax(input: &[f32], rows: usize, width: usize) -> anyhow::Result<Vec<f32>> {
    if input.len() != rows * width || width == 0 {
        anyhow::bail!("invalid decoder log-softmax dimensions")
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * width;
        let maximum = input[offset..offset + width]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = numerics::decoder_log_softmax_denominator(
            &input[offset..offset + width],
            maximum,
        );
        let normalizer = maximum + denominator.ln();
        for index in 0..width {
            output[offset + index] = input[offset + index] - normalizer;
        }
    }
    Ok(output)
}

pub fn decoder_reduce_sum(input: &[f32], rows: usize, width: usize) -> anyhow::Result<Vec<f32>> {
    if input.len() != rows * width {
        anyhow::bail!("invalid reduce-sum dimensions")
    }
    Ok((0..rows)
        .map(|row| input[row * width..(row + 1) * width].iter().sum())
        .collect())
}

pub fn original_add(
    left_shape: &[usize],
    left: &[f16],
    right_shape: &[usize],
    right: &[f16],
) -> anyhow::Result<(Vec<usize>, Vec<f16>)> {
    broadcast_binary(left_shape, left, right_shape, right, rounded_add)
}

pub fn original_multiply(
    left_shape: &[usize],
    left: &[f16],
    right_shape: &[usize],
    right: &[f16],
) -> anyhow::Result<(Vec<usize>, Vec<f16>)> {
    broadcast_binary(left_shape, left, right_shape, right, rounded_mul)
}

fn broadcast_binary<F>(
    raw_left_shape: &[usize],
    left: &[f16],
    raw_right_shape: &[usize],
    right: &[f16],
    operation: F,
) -> anyhow::Result<(Vec<usize>, Vec<f16>)>
where
    F: Fn(f16, f16) -> f16,
{
    if shape_elements(raw_left_shape)? != left.len()
        || shape_elements(raw_right_shape)? != right.len()
    {
        anyhow::bail!("broadcast input shape does not match data length")
    }
    let rank = raw_left_shape.len().max(raw_right_shape.len());
    let mut left_shape = vec![1; rank - raw_left_shape.len()];
    left_shape.extend_from_slice(raw_left_shape);
    let mut right_shape = vec![1; rank - raw_right_shape.len()];
    right_shape.extend_from_slice(raw_right_shape);
    let mut output_shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left_dimension = left_shape[axis];
        let right_dimension = right_shape[axis];
        if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
            anyhow::bail!("broadcast input shapes are incompatible")
        }
        output_shape.push(if left_dimension == 1 {
            right_dimension
        } else {
            left_dimension
        });
    }
    let left_strides = contiguous_strides(&left_shape)?;
    let right_strides = contiguous_strides(&right_shape)?;
    let output_elements = shape_elements(&output_shape)?;
    let mut output = Vec::with_capacity(output_elements);
    for linear in 0..output_elements {
        let mut remainder = linear;
        let mut left_offset = 0;
        let mut right_offset = 0;
        for axis in (0..rank).rev() {
            let coordinate = remainder % output_shape[axis];
            remainder /= output_shape[axis];
            if left_shape[axis] != 1 {
                left_offset += coordinate * left_strides[axis];
            }
            if right_shape[axis] != 1 {
                right_offset += coordinate * right_strides[axis];
            }
        }
        output.push(operation(left[left_offset], right[right_offset]));
    }
    Ok((output_shape, output))
}

fn shape_elements(shape: &[usize]) -> anyhow::Result<usize> {
    shape.iter().try_fold(1usize, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| anyhow::anyhow!("tensor shape overflows element count"))
    })
}

fn contiguous_strides(shape: &[usize]) -> anyhow::Result<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index]
            .checked_mul(shape[index])
            .ok_or_else(|| anyhow::anyhow!("tensor stride overflows element count"))?;
    }
    Ok(strides)
}

pub fn gelu_f16(input: &[f16]) -> Vec<f16> {
    input.iter().copied().map(numerics::gelu).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn standard_conv(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    batch: usize,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_height: usize,
    stride_width: usize,
) -> anyhow::Result<Vec<f16>> {
    if input.len() != batch * input_channels * input_height * input_width
        || weights.len() != output_channels * input_channels * kernel_height * kernel_width
        || bias.len() != output_channels
        || (input_channels != 1 && !input_channels.is_multiple_of(8))
        || kernel_height == 0
        || kernel_width == 0
        || kernel_height > input_height
        || kernel_width > input_width
        || stride_height == 0
        || stride_width == 0
    {
        anyhow::bail!("invalid convolution dimensions")
    }
    let output_height = (input_height - kernel_height) / stride_height + 1;
    let output_width = (input_width - kernel_width) / stride_width + 1;
    let mut output = vec![f16::from_f32(0.0); batch * output_channels * output_height * output_width];
    for batch_index in 0..batch {
        for (channel, bias_value) in bias.iter().copied().enumerate() {
            for y in 0..output_height {
                for x in 0..output_width {
                    let mut stash = 0.0_f32;
                    let channel_group = if input_channels == 1 { 1 } else { 8 };
                    for channel_block in (0..input_channels).step_by(channel_group) {
                        let mut group = [f16::from_f32(0.0); 8];
                        for kernel_y in 0..kernel_height {
                            for kernel_x in 0..kernel_width {
                                for (lane, accumulator) in
                                    group.iter_mut().enumerate().take(channel_group)
                                {
                                    let input_channel = channel_block + lane;
                                    let input_index = (((batch_index * input_channels
                                        + input_channel)
                                        * input_height
                                        + y * stride_height
                                        + kernel_y)
                                        * input_width)
                                        + x * stride_width
                                        + kernel_x;
                                    let weight_index = (((channel * input_channels
                                        + input_channel)
                                        * kernel_height
                                        + kernel_y)
                                        * kernel_width)
                                        + kernel_x;
                                    *accumulator = rounded_add(
                                        rounded_mul(input[input_index], weights[weight_index]),
                                        *accumulator,
                                    );
                                }
                            }
                        }
                        for lane in (0..channel_group).rev() {
                            stash += group[lane].to_f32();
                        }
                    }
                    let output_index = (((batch_index * output_channels + channel) * output_height + y) * output_width) + x;
                    output[output_index] = rounded_add(f16::from_f32(stash), bias_value);
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
) -> anyhow::Result<Vec<f16>> {
    if input.len() != batch * channels * input_height * input_width
        || weights.len() != channels * kernel_height * kernel_width
        || bias.len() != channels
        || kernel_height > input_height
        || kernel_width > input_width
    {
        anyhow::bail!("invalid depthwise convolution dimensions")
    }
    let output_height = input_height - kernel_height + 1;
    let output_width = input_width - kernel_width + 1;
    let mut output = vec![
        f16::from_f32(0.0);
        batch * channels * output_height * output_width
    ];
    for batch_index in 0..batch {
        for (channel, bias_value) in bias.iter().copied().enumerate() {
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let mut accumulator = bias_value;
                    for kernel_y in 0..kernel_height {
                        for kernel_x in 0..kernel_width {
                            let input_index = (((batch_index * channels + channel)
                                * input_height
                                + output_y
                                + kernel_y)
                                * input_width)
                                + output_x
                                + kernel_x;
                            let weight_index = (channel * kernel_height + kernel_y)
                                * kernel_width
                                + kernel_x;
                            accumulator = rounded_add(
                                rounded_mul(input[input_index], weights[weight_index]),
                                accumulator,
                            );
                        }
                    }
                    let output_index = (((batch_index * channels + channel)
                        * output_height
                        + output_y)
                        * output_width)
                        + output_x;
                    output[output_index] = accumulator;
                }
            }
        }
    }
    Ok(output)
}

pub fn punctuation_qk(query: &[f32], key: &[f32], batches: usize, rows: usize, inner: usize) -> anyhow::Result<Vec<f32>> {
    if query.len() != batches * rows * inner || key.len() != batches * rows * inner {
        anyhow::bail!("invalid punctuation QK dimensions")
    }
    let mut output = vec![0.0; batches * rows * rows];
    for batch in 0..batches {
        for row in 0..rows {
            for column in 0..rows {
                let mut sum = 0.0;
                for index in 0..inner {
                    sum += query[(batch * rows + row) * inner + index]
                        * key[(batch * rows + column) * inner + index];
                }
                output[(batch * rows + row) * rows + column] = sum;
            }
        }
    }
    Ok(output)
}

pub fn punctuation_context(probabilities: &[f32], value: &[f32], batches: usize, rows: usize, inner: usize, columns: usize) -> anyhow::Result<Vec<f32>> {
    if probabilities.len() != batches * rows * inner || value.len() != batches * inner * columns {
        anyhow::bail!("invalid punctuation context dimensions")
    }
    let mut output = vec![0.0; batches * rows * columns];
    for batch in 0..batches {
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0;
                for index in 0..inner {
                    sum += probabilities[(batch * rows + row) * inner + index]
                        * value[(batch * inner + index) * columns + column];
                }
                output[(batch * rows + row) * columns + column] = sum;
            }
        }
    }
    Ok(output)
}

pub fn punctuation_quantized_linear(input: &[i8], weights: &[i8], rows: usize, columns: usize, inner: usize) -> anyhow::Result<Vec<i32>> {
    if input.len() != rows * inner || weights.len() != columns * inner {
        anyhow::bail!("invalid punctuation quantized linear dimensions")
    }
    let mut output = vec![0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0;
            for index in 0..inner {
                sum += input[row * inner + index] as i32 * weights[column * inner + index] as i32;
            }
            output[row * columns + column] = sum;
        }
    }
    Ok(output)
}

pub fn punctuation_softmax(input: &[f32], rows: usize, columns: usize) -> anyhow::Result<Vec<f32>> {
    if input.len() != rows * columns || columns == 0 {
        anyhow::bail!("invalid punctuation softmax dimensions")
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * columns;
        let maximum = input[offset..offset + columns]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = input[offset..offset + columns]
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f32>();
        for column in 0..columns {
            output[offset + column] = (input[offset + column] - maximum).exp() / denominator;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{
        decoder_active_rows, decoder_matmul, matmul_f16, original_add,
        set_decoder_active_rows, standard_conv,
    };

    #[test]
    fn original_add_supports_multidimensional_broadcast_with_half_rounding() {
        let left = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let right = [f16::from_f32(0.5), f16::from_f32(-0.5)];
        let (shape, output) = original_add(&[2, 1, 3], &left, &[1, 2, 1], &right).unwrap();
        assert_eq!(shape, [2, 2, 3]);
        assert_eq!(
            output,
            [
                f16::from_f32(1.5),
                f16::from_f32(2.5),
                f16::from_f32(3.5),
                f16::from_f32(0.5),
                f16::from_f32(1.5),
                f16::from_f32(2.5),
                f16::from_f32(4.5),
                f16::from_f32(5.5),
                f16::from_f32(6.5),
                f16::from_f32(3.5),
                f16::from_f32(4.5),
                f16::from_f32(5.5),
            ]
        );
    }

    #[test]
    fn matmul_rejects_shape_mismatches() {
        let left = [f16::ONE; 3];
        let right = [f16::ONE; 4];
        assert!(matmul_f16(&left, &right, 2, 2, 2).is_err());
    }

    #[test]
    fn decoder_matmul_keeps_vendor_column_reduction_order() {
        let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let right = [
            0.5_f32, 1.0, -0.5,
            1.5, 2.0, -1.5,
            2.5, 3.0, -2.5,
            3.5, 4.0, -3.5,
            4.5, 5.0, -4.5,
        ];
        let output = decoder_matmul(&left, &right, 1, 5, 3).unwrap();
        assert_eq!(output, [47.5, 55.0, -47.5]);
    }

    #[test]
    fn convolution_accumulates_channel_groups_before_bias() {
        let input = [f16::ONE; 8];
        let weights = [f16::from_f32(0.5); 8];
        let bias = [f16::from_f32(0.25)];
        let output = standard_conv(
            &input, &weights, &bias, 1, 8, 1, 1, 1, 1, 1, 1, 1,
        )
        .unwrap();
        assert_eq!(output, [f16::from_f32(4.25)]);
    }

    #[test]
    fn decoder_active_rows_rejects_invalid_values() {
        set_decoder_active_rows(3).unwrap();
        assert_eq!(decoder_active_rows(), 3);
        assert!(set_decoder_active_rows(0).is_err());
        assert!(set_decoder_active_rows(-1).is_err());
        set_decoder_active_rows(1).unwrap();
    }
}
