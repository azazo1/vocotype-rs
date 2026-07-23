use std::cell::RefCell;

use anyhow::Result;
use half::f16;

use super::Conv2dParams;
use super::neon;

#[derive(Default)]
struct AccelerateWorkspace {
    left: Vec<f32>,
    right: Vec<f32>,
    patches: Vec<f32>,
    result: Vec<f32>,
}

thread_local! {
    static ACCELERATE_WORKSPACE: RefCell<AccelerateWorkspace> = RefCell::new(AccelerateWorkspace::default());
}

pub(crate) const fn available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

pub(crate) fn gemm_f16_into(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
    output: &mut [f16],
) -> Result<()> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let rows_i32 = cblas_dimension(rows)?;
        let inner_i32 = cblas_dimension(inner)?;
        let columns_i32 = cblas_dimension(columns)?;
        ACCELERATE_WORKSPACE.with_borrow_mut(|workspace| {
            workspace.left.resize(left.len(), 0.0);
            workspace.right.resize(right.len(), 0.0);
            workspace.result.resize(output.len(), 0.0);
            neon::convert_half_to_float(left, &mut workspace.left);
            neon::convert_half_to_float(right, &mut workspace.right);
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    rows_i32,
                    columns_i32,
                    inner_i32,
                    1.0,
                    workspace.left.as_ptr(),
                    inner_i32,
                    workspace.right.as_ptr(),
                    inner_i32,
                    0.0,
                    workspace.result.as_mut_ptr(),
                    columns_i32,
                );
            }
            for row in 0..rows {
                for (column, bias_value) in bias.iter().enumerate() {
                    workspace.result[row * columns + column] += bias_value.to_f32();
                }
            }
            neon::convert_float_to_half(&workspace.result[..output.len()], output);
        });
        Ok(())
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (left, right, bias, rows, inner, columns, output);
        anyhow::bail!("Apple Accelerate is unavailable on this platform")
    }
}

pub(crate) fn standard_conv_f16_into(
    input: &[f16],
    weights: &[f16],
    bias: &[f16],
    params: Conv2dParams,
    output: &mut [f16],
) -> Result<()> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let output_height = params.output_height();
        let output_width = params.output_width();
        let rows = params.batch * output_height * output_width;
        let inner = params.input_channels * params.kernel_height * params.kernel_width;
        let rows_i32 = cblas_dimension(rows)?;
        let inner_i32 = cblas_dimension(inner)?;
        let output_channels_i32 = cblas_dimension(params.output_channels)?;
        ACCELERATE_WORKSPACE.with_borrow_mut(|workspace| {
            workspace.left.resize(input.len(), 0.0);
            workspace.right.resize(weights.len(), 0.0);
            workspace.patches.resize(rows * inner, 0.0);
            workspace
                .result
                .resize(rows * params.output_channels, 0.0);
            neon::convert_half_to_float(input, &mut workspace.left);
            neon::convert_half_to_float(weights, &mut workspace.right);

            for batch in 0..params.batch {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        let row = (batch * output_height + output_y) * output_width + output_x;
                        let mut patch_index = row * inner;
                        for channel in 0..params.input_channels {
                            for kernel_y in 0..params.kernel_height {
                                let input_y = output_y * params.stride_height + kernel_y;
                                for kernel_x in 0..params.kernel_width {
                                    let input_x = output_x * params.stride_width + kernel_x;
                                    let input_index = ((batch * params.input_channels + channel)
                                        * params.input_height
                                        + input_y)
                                        * params.input_width
                                        + input_x;
                                    workspace.patches[patch_index] = workspace.left[input_index];
                                    patch_index += 1;
                                }
                            }
                        }
                    }
                }
            }

            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    rows_i32,
                    output_channels_i32,
                    inner_i32,
                    1.0,
                    workspace.patches.as_ptr(),
                    inner_i32,
                    workspace.right.as_ptr(),
                    inner_i32,
                    0.0,
                    workspace.result.as_mut_ptr(),
                    output_channels_i32,
                );
            }

            for batch in 0..params.batch {
                for (channel, bias_value) in bias.iter().enumerate() {
                    let bias_value = bias_value.to_f32();
                    for output_y in 0..output_height {
                        for output_x in 0..output_width {
                            let row = (batch * output_height + output_y) * output_width + output_x;
                            let output_index = ((batch * params.output_channels + channel)
                                * output_height
                                + output_y)
                                * output_width
                                + output_x;
                            output[output_index] = f16::from_f32(
                                workspace.result[row * params.output_channels + channel]
                                    + bias_value,
                            );
                        }
                    }
                }
            }
        });
        Ok(())
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (input, weights, bias, params, output);
        anyhow::bail!("Apple Accelerate is unavailable on this platform")
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn cblas_dimension(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("matrix dimension exceeds Accelerate limits"))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const CBLAS_TRANS: i32 = 112;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        rows: i32,
        columns: i32,
        inner: i32,
        alpha: f32,
        left: *const f32,
        left_stride: i32,
        right: *const f32,
        right_stride: i32,
        beta: f32,
        output: *mut f32,
        output_stride: i32,
    );
}
