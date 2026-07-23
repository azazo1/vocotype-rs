use std::sync::atomic::{AtomicI32, Ordering};

use half::f16;

use crate::numerics;

mod accelerate;
mod decoder_gemm;
mod neon;

pub use decoder_gemm::{
    DecoderGemmBlock, DecoderGemmParams, decoder_gemm_column_block, decoder_gemm_element,
    decoder_gemm_into, decoder_gemm_params, pack_decoder_gemm_left,
    pack_decoder_gemm_left_into,
};

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

pub const fn optimized_kernel_backend() -> &'static str {
    if accelerate::available() {
        "accelerate-neon"
    } else if neon::available() {
        "neon"
    } else {
        "portable"
    }
}

pub fn gemm_f16_output_len(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
) -> anyhow::Result<usize> {
    let left_len = rows
        .checked_mul(inner)
        .ok_or_else(|| anyhow::anyhow!("GEMM left dimensions overflow"))?;
    let right_len = columns
        .checked_mul(inner)
        .ok_or_else(|| anyhow::anyhow!("GEMM right dimensions overflow"))?;
    let output_len = rows
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("GEMM output dimensions overflow"))?;
    if rows == 0
        || inner == 0
        || columns == 0
        || left.len() != left_len
        || right.len() != right_len
        || bias.len() != columns
    {
        anyhow::bail!("invalid GEMM dimensions")
    }
    Ok(output_len)
}

pub const fn gemm_f16_uses_accelerate(rows: usize) -> bool {
    accelerate::available() && rows >= 8
}

pub const fn gemm_f16_uses_packed_neon(rows: usize) -> bool {
    !accelerate::available() && neon::available() && rows >= 8 && rows.is_multiple_of(8)
}

pub fn gemm_f16_element(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    inner: usize,
    columns: usize,
    linear: usize,
) -> f16 {
    let row = linear / columns;
    let column = linear % columns;
    let mut value = bias[column].to_f32();
    for index in 0..inner {
        value = left[row * inner + index]
            .to_f32()
            .mul_add(right[column * inner + index].to_f32(), value);
    }
    f16::from_f32(value)
}

pub fn gemm_f16_into(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
    output: &mut [f16],
) -> anyhow::Result<()> {
    let output_len = gemm_f16_output_len(left, right, bias, rows, inner, columns)?;
    if output.len() != output_len {
        anyhow::bail!("invalid GEMM output dimensions")
    }
    if gemm_f16_uses_accelerate(rows) {
        return accelerate::gemm_f16_into(
            left, right, bias, rows, inner, columns, output,
        );
    }
    if gemm_f16_uses_packed_neon(rows)
        && neon::packed_gemm_f16_into(left, right, bias, rows, inner, columns, output)
    {
        return Ok(());
    }
    for (linear, target) in output.iter_mut().enumerate() {
        *target = gemm_f16_element(left, right, bias, inner, columns, linear);
    }
    Ok(())
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dParams {
    pub batch: usize,
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_channels: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride_height: usize,
    pub stride_width: usize,
}

impl Conv2dParams {
    pub const fn output_height(self) -> usize {
        (self.input_height - self.kernel_height) / self.stride_height + 1
    }

    pub const fn output_width(self) -> usize {
        (self.input_width - self.kernel_width) / self.stride_width + 1
    }
}

pub fn standard_conv_output_len(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
) -> anyhow::Result<usize> {
    let input_len = params
        .batch
        .checked_mul(params.input_channels)
        .and_then(|value| value.checked_mul(params.input_height))
        .and_then(|value| value.checked_mul(params.input_width))
        .ok_or_else(|| anyhow::anyhow!("convolution input dimensions overflow"))?;
    let weight_len = params
        .output_channels
        .checked_mul(params.input_channels)
        .and_then(|value| value.checked_mul(params.kernel_height))
        .and_then(|value| value.checked_mul(params.kernel_width))
        .ok_or_else(|| anyhow::anyhow!("convolution weight dimensions overflow"))?;
    if params.batch == 0
        || params.input_channels == 0
        || params.input_height == 0
        || params.input_width == 0
        || params.output_channels == 0
        || params.kernel_height == 0
        || params.kernel_width == 0
        || params.kernel_height > params.input_height
        || params.kernel_width > params.input_width
        || params.stride_height == 0
        || params.stride_width == 0
        || (params.input_channels != 1 && !params.input_channels.is_multiple_of(8))
        || input.len() != input_len
        || weights.len() != weight_len
        || bias.len() != params.output_channels
    {
        anyhow::bail!("invalid convolution dimensions")
    }
    params
        .batch
        .checked_mul(params.output_channels)
        .and_then(|value| value.checked_mul(params.output_height()))
        .and_then(|value| value.checked_mul(params.output_width()))
        .ok_or_else(|| anyhow::anyhow!("convolution output dimensions overflow"))
}

pub const fn standard_conv_uses_accelerate(params: Conv2dParams) -> bool {
    accelerate::available() && params.input_channels != 1
}

pub fn standard_conv_element(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
    linear: usize,
) -> f16 {
    let output_height = params.output_height();
    let output_width = params.output_width();
    let output_x = linear % output_width;
    let output_y = (linear / output_width) % output_height;
    let output_channel =
        (linear / (output_width * output_height)) % params.output_channels;
    let batch = linear / (output_width * output_height * params.output_channels);
    let mut stash = 0.0_f32;
    let channel_group = if params.input_channels == 1 { 1 } else { 8 };
    for channel_block in (0..params.input_channels).step_by(channel_group) {
        let mut group = [f16::ZERO; 8];
        for kernel_y in 0..params.kernel_height {
            for kernel_x in 0..params.kernel_width {
                for (lane, accumulator) in group.iter_mut().enumerate().take(channel_group) {
                    let input_channel = channel_block + lane;
                    let input_y = output_y * params.stride_height + kernel_y;
                    let input_x = output_x * params.stride_width + kernel_x;
                    let input_index = ((batch * params.input_channels + input_channel)
                        * params.input_height
                        + input_y)
                        * params.input_width
                        + input_x;
                    let weight_index = ((output_channel * params.input_channels + input_channel)
                        * params.kernel_height
                        + kernel_y)
                        * params.kernel_width
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
    rounded_add(f16::from_f32(stash), bias[output_channel])
}

pub fn standard_conv_into(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
    output: &mut [f16],
) -> anyhow::Result<()> {
    let output_len = standard_conv_output_len(input, weights, bias, params)?;
    if output.len() != output_len {
        anyhow::bail!("invalid convolution output dimensions")
    }
    if standard_conv_uses_accelerate(params) {
        return accelerate::standard_conv_f16_into(input, weights, bias, params, output);
    }
    for (linear, target) in output.iter_mut().enumerate() {
        *target = standard_conv_element(input, weights, bias, params, linear);
    }
    Ok(())
}

pub fn depthwise_conv_output_len(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
) -> anyhow::Result<usize> {
    if params.output_channels != params.input_channels
        || params.stride_height != 1
        || params.stride_width != 1
    {
        anyhow::bail!("invalid depthwise convolution parameters")
    }
    let input_len = params
        .batch
        .checked_mul(params.input_channels)
        .and_then(|value| value.checked_mul(params.input_height))
        .and_then(|value| value.checked_mul(params.input_width))
        .ok_or_else(|| anyhow::anyhow!("depthwise convolution input dimensions overflow"))?;
    let weight_len = params
        .input_channels
        .checked_mul(params.kernel_height)
        .and_then(|value| value.checked_mul(params.kernel_width))
        .ok_or_else(|| anyhow::anyhow!("depthwise convolution weight dimensions overflow"))?;
    if params.batch == 0
        || params.input_channels == 0
        || params.input_height == 0
        || params.input_width == 0
        || params.kernel_height == 0
        || params.kernel_width == 0
        || params.kernel_height > params.input_height
        || params.kernel_width > params.input_width
        || input.len() != input_len
        || weights.len() != weight_len
        || bias.len() != params.input_channels
    {
        anyhow::bail!("invalid depthwise convolution dimensions")
    }
    params
        .batch
        .checked_mul(params.input_channels)
        .and_then(|value| value.checked_mul(params.output_height()))
        .and_then(|value| value.checked_mul(params.output_width()))
        .ok_or_else(|| anyhow::anyhow!("depthwise convolution output dimensions overflow"))
}

pub fn depthwise_conv_element(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
    linear: usize,
) -> f16 {
    let output_height = params.output_height();
    let output_width = params.output_width();
    let output_x = linear % output_width;
    let output_y = (linear / output_width) % output_height;
    let channel = (linear / (output_width * output_height)) % params.input_channels;
    let batch = linear / (output_width * output_height * params.input_channels);
    let mut accumulator = bias[channel];
    for kernel_y in 0..params.kernel_height {
        for kernel_x in 0..params.kernel_width {
            let input_index = ((batch * params.input_channels + channel) * params.input_height
                + output_y
                + kernel_y)
                * params.input_width
                + output_x
                + kernel_x;
            let weight_index = (channel * params.kernel_height + kernel_y)
                * params.kernel_width
                + kernel_x;
            accumulator = rounded_add(
                rounded_mul(input[input_index], weights[weight_index]),
                accumulator,
            );
        }
    }
    accumulator
}

pub fn depthwise_conv_into(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
    output: &mut [f16],
) -> anyhow::Result<()> {
    let output_len = depthwise_conv_output_len(input, weights, bias, params)?;
    if output.len() != output_len {
        anyhow::bail!("invalid depthwise convolution output dimensions")
    }
    for (linear, target) in output.iter_mut().enumerate() {
        *target = depthwise_conv_element(input, weights, bias, params, linear);
    }
    Ok(())
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
    use std::sync::Mutex;

    use half::f16;

    use super::{
        Conv2dParams, decoder_active_rows, decoder_gemm_element, decoder_gemm_into,
        decoder_gemm_params, decoder_matmul, gemm_f16_element, gemm_f16_into, matmul_f16,
        original_add, set_decoder_active_rows, standard_conv_element, standard_conv_into,
    };

    static DECODER_ACTIVE_ROWS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn patterned_f16(length: usize, offset: usize) -> Vec<f16> {
        (0..length)
            .map(|index| {
                let value = ((index * 17 + offset) % 29) as f32 - 14.0;
                f16::from_f32(value / 32.0)
            })
            .collect()
    }

    fn assert_f16_close(actual: &[f16], expected: &[f16], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual = actual.to_f32();
            let expected = expected.to_f32();
            assert!(actual.is_finite(), "output {index} is not finite");
            assert!(
                (actual - expected).abs() <= tolerance,
                "output {index} differs: actual={actual}, expected={expected}"
            );
        }
    }

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
    fn decoder_gemm_neon_matches_scalar_for_active_rows() {
        let _guard = DECODER_ACTIVE_ROWS_TEST_LOCK.lock().unwrap();
        let rows = 8;
        let inner = 12;
        let columns = 9;
        let left = patterned_f16(rows * inner, 3);
        let right = patterned_f16(columns * inner, 7);
        let bias = patterned_f16(columns, 11);
        for active_rows in [1, 8] {
            set_decoder_active_rows(active_rows).unwrap();
            let params = decoder_gemm_params(
                &left,
                &right,
                &bias,
                rows,
                inner,
                columns,
            )
            .unwrap();
            let expected = (0..params.output_len())
                .map(|linear| decoder_gemm_element(&left, &right, &bias, params, linear))
                .collect::<Vec<_>>();
            let mut actual = vec![f16::ZERO; params.output_len()];
            decoder_gemm_into(
                &left,
                &right,
                &bias,
                rows,
                inner,
                columns,
                &mut actual,
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
        set_decoder_active_rows(1).unwrap();
    }

    #[test]
    fn convolution_accumulates_channel_groups_before_bias() {
        let input = [f16::ONE; 8];
        let weights = [f16::from_f32(0.5); 8];
        let bias = [f16::from_f32(0.25)];
        let mut output = [f16::ZERO; 1];
        standard_conv_into(
            &input,
            &weights,
            &bias,
            Conv2dParams {
                batch: 1,
                input_channels: 8,
                input_height: 1,
                input_width: 1,
                output_channels: 1,
                kernel_height: 1,
                kernel_width: 1,
                stride_height: 1,
                stride_width: 1,
            },
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [f16::from_f32(4.25)]);
    }

    #[test]
    fn optimized_gemm_matches_reference_for_non_multiple_row_count() {
        let rows = 9;
        let inner = 13;
        let columns = 7;
        let left = patterned_f16(rows * inner, 3);
        let right = patterned_f16(columns * inner, 7);
        let bias = patterned_f16(columns, 11);
        let expected = (0..rows * columns)
            .map(|linear| {
                gemm_f16_element(&left, &right, &bias, inner, columns, linear)
            })
            .collect::<Vec<_>>();
        let mut actual = vec![f16::ZERO; expected.len()];

        gemm_f16_into(
            &left,
            &right,
            &bias,
            rows,
            inner,
            columns,
            &mut actual,
        )
        .unwrap();

        assert_f16_close(&actual, &expected, 0.004);
    }

    #[test]
    fn im2col_convolution_matches_reference() {
        let params = Conv2dParams {
            batch: 2,
            input_channels: 8,
            input_height: 4,
            input_width: 5,
            output_channels: 3,
            kernel_height: 2,
            kernel_width: 3,
            stride_height: 2,
            stride_width: 1,
        };
        let input = patterned_f16(
            params.batch * params.input_channels * params.input_height * params.input_width,
            5,
        );
        let weights = patterned_f16(
            params.output_channels
                * params.input_channels
                * params.kernel_height
                * params.kernel_width,
            13,
        );
        let bias = patterned_f16(params.output_channels, 19);
        let output_len = params.batch
            * params.output_channels
            * params.output_height()
            * params.output_width();
        let expected = (0..output_len)
            .map(|linear| standard_conv_element(&input, &weights, &bias, params, linear))
            .collect::<Vec<_>>();
        let mut actual = vec![f16::ZERO; output_len];

        standard_conv_into(&input, &weights, &bias, params, &mut actual).unwrap();

        assert_f16_close(&actual, &expected, 0.05);
    }

    #[test]
    fn single_channel_convolution_keeps_reference_order() {
        let params = Conv2dParams {
            batch: 1,
            input_channels: 1,
            input_height: 4,
            input_width: 5,
            output_channels: 2,
            kernel_height: 3,
            kernel_width: 2,
            stride_height: 1,
            stride_width: 2,
        };
        let input = patterned_f16(params.input_height * params.input_width, 2);
        let weights = patterned_f16(
            params.output_channels * params.kernel_height * params.kernel_width,
            17,
        );
        let bias = patterned_f16(params.output_channels, 23);
        let output_len = params.output_channels * params.output_height() * params.output_width();
        let expected = (0..output_len)
            .map(|linear| standard_conv_element(&input, &weights, &bias, params, linear))
            .collect::<Vec<_>>();
        let mut actual = vec![f16::ZERO; output_len];

        standard_conv_into(&input, &weights, &bias, params, &mut actual).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn decoder_active_rows_rejects_invalid_values() {
        let _guard = DECODER_ACTIVE_ROWS_TEST_LOCK.lock().unwrap();
        set_decoder_active_rows(3).unwrap();
        assert_eq!(decoder_active_rows(), 3);
        assert!(set_decoder_active_rows(0).is_err());
        assert!(set_decoder_active_rows(-1).is_err());
        set_decoder_active_rows(1).unwrap();
    }
}
