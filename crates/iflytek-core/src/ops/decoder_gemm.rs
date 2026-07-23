use half::f16;

use super::{decoder_active_rows, rounded_add};

const ROWS_PER_BLOCK: usize = 8;
const COLUMNS_PER_BLOCK: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderGemmParams {
    rows: usize,
    inner: usize,
    columns: usize,
    paired: bool,
}

impl DecoderGemmParams {
    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn inner(self) -> usize {
        self.inner
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn output_len(self) -> usize {
        self.rows * self.columns
    }

    pub const fn column_blocks(self) -> usize {
        self.columns.div_ceil(COLUMNS_PER_BLOCK)
    }

    pub const fn uses_packed_neon(self) -> bool {
        self.rows == ROWS_PER_BLOCK && cfg!(target_arch = "aarch64")
    }
}

pub struct DecoderGemmBlock {
    values: [[f16; ROWS_PER_BLOCK]; COLUMNS_PER_BLOCK],
    columns: usize,
}

impl DecoderGemmBlock {
    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn value(&self, column: usize, row: usize) -> f16 {
        self.values[column][row]
    }
}

pub fn decoder_gemm_params(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
) -> anyhow::Result<DecoderGemmParams> {
    let left_len = rows
        .checked_mul(inner)
        .ok_or_else(|| anyhow::anyhow!("decoder GEMM left dimensions overflow"))?;
    let right_len = columns
        .checked_mul(inner)
        .ok_or_else(|| anyhow::anyhow!("decoder GEMM right dimensions overflow"))?;
    rows.checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("decoder GEMM output dimensions overflow"))?;
    let active = usize::try_from(decoder_active_rows())
        .map_err(|_| anyhow::anyhow!("decoder active rows must be positive"))?;
    if rows == 0
        || inner == 0
        || columns == 0
        || left.len() != left_len
        || right.len() != right_len
        || bias.len() != columns
        || active == 0
        || active > rows
    {
        anyhow::bail!("invalid decoder GEMM dimensions")
    }
    let paired = active == 1;
    if paired && !inner.is_multiple_of(4) {
        anyhow::bail!("one-row decoder GEMM requires an inner multiple of four")
    }
    Ok(DecoderGemmParams {
        rows,
        inner,
        columns,
        paired,
    })
}

pub fn pack_decoder_gemm_left(left: &[f16], params: DecoderGemmParams) -> Vec<f16> {
    let mut packed = Vec::new();
    pack_decoder_gemm_left_into(left, params, &mut packed);
    packed
}

pub fn pack_decoder_gemm_left_into(
    left: &[f16],
    params: DecoderGemmParams,
    packed: &mut Vec<f16>,
) {
    if params.rows != ROWS_PER_BLOCK {
        packed.clear();
        return;
    }
    packed.resize(left.len(), f16::ZERO);
    for index in 0..params.inner {
        for row in 0..ROWS_PER_BLOCK {
            packed[index * ROWS_PER_BLOCK + row] = left[row * params.inner + index];
        }
    }
}

pub fn decoder_gemm_element(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    params: DecoderGemmParams,
    linear: usize,
) -> f16 {
    let row = linear / params.columns;
    let column = linear % params.columns;
    let values = &left[row * params.inner..(row + 1) * params.inner];
    let weights = &right[column * params.inner..(column + 1) * params.inner];
    let mut even = bias[column];
    let mut odd = f16::ZERO;
    for index in 0..params.inner {
        let accumulator = if params.paired && index & 1 != 0 {
            &mut odd
        } else {
            &mut even
        };
        *accumulator = f16::from_f32(
            values[index]
                .to_f32()
                .mul_add(weights[index].to_f32(), accumulator.to_f32()),
        );
    }
    if params.paired {
        rounded_add(even, odd)
    } else {
        even
    }
}

pub fn decoder_gemm_column_block(
    packed_left: &[f16],
    right: &[f16],
    bias: &[f16],
    params: DecoderGemmParams,
    block: usize,
) -> DecoderGemmBlock {
    assert_eq!(params.rows, ROWS_PER_BLOCK);
    assert_eq!(packed_left.len(), params.rows * params.inner);
    let first_column = block * COLUMNS_PER_BLOCK;
    let columns = COLUMNS_PER_BLOCK.min(params.columns - first_column);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decoder_gemm_column_block_neon(
                packed_left,
                right,
                bias,
                params,
                first_column,
                columns,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut values = [[f16::ZERO; ROWS_PER_BLOCK]; COLUMNS_PER_BLOCK];
        for column in 0..columns {
            let absolute_column = first_column + column;
            for row in 0..ROWS_PER_BLOCK {
                values[column][row] = decoder_gemm_element(
                    &unpack_decoder_gemm_left(packed_left, params),
                    right,
                    bias,
                    params,
                    row * params.columns + absolute_column,
                );
            }
        }
        DecoderGemmBlock { values, columns }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn decoder_gemm_column_block_neon(
    packed_left: &[f16],
    right: &[f16],
    bias: &[f16],
    params: DecoderGemmParams,
    first_column: usize,
    columns: usize,
) -> DecoderGemmBlock {
    use core::arch::aarch64::{
        vaddq_f16, vdupq_n_u16, vfmaq_f16, vld1q_u16, vreinterpretq_f16_u16,
        vreinterpretq_u16_f16, vst1q_u16,
    };

    let zero = vreinterpretq_f16_u16(vdupq_n_u16(0));
    let mut even = [zero; COLUMNS_PER_BLOCK];
    let mut odd = [zero; COLUMNS_PER_BLOCK];
    for column in 0..columns {
        even[column] = vreinterpretq_f16_u16(vdupq_n_u16(
            bias[first_column + column].to_bits(),
        ));
    }
    for index in 0..params.inner {
        let left = vreinterpretq_f16_u16(unsafe {
            vld1q_u16(
                packed_left
                    .as_ptr()
                    .add(index * ROWS_PER_BLOCK)
                    .cast::<u16>(),
            )
        });
        for column in 0..columns {
            let weight = right[(first_column + column) * params.inner + index].to_bits();
            let weight = vreinterpretq_f16_u16(vdupq_n_u16(weight));
            if params.paired && index & 1 != 0 {
                odd[column] = vfmaq_f16(odd[column], left, weight);
            } else {
                even[column] = vfmaq_f16(even[column], left, weight);
            }
        }
    }

    let mut values = [[f16::ZERO; ROWS_PER_BLOCK]; COLUMNS_PER_BLOCK];
    for column in 0..columns {
        if params.paired {
            even[column] = vaddq_f16(even[column], odd[column]);
        }
        unsafe {
            vst1q_u16(
                values[column].as_mut_ptr().cast::<u16>(),
                vreinterpretq_u16_f16(even[column]),
            );
        }
    }
    DecoderGemmBlock { values, columns }
}

#[cfg(not(target_arch = "aarch64"))]
fn unpack_decoder_gemm_left(packed: &[f16], params: DecoderGemmParams) -> Vec<f16> {
    let mut left = vec![f16::ZERO; packed.len()];
    for index in 0..params.inner {
        for row in 0..params.rows {
            left[row * params.inner + index] = packed[index * params.rows + row];
        }
    }
    left
}

pub fn decoder_gemm_into(
    left: &[f16],
    right: &[f16],
    bias: &[f16],
    rows: usize,
    inner: usize,
    columns: usize,
    output: &mut [f16],
) -> anyhow::Result<()> {
    let params = decoder_gemm_params(left, right, bias, rows, inner, columns)?;
    if output.len() != params.output_len() {
        anyhow::bail!("invalid decoder GEMM output dimensions")
    }
    if params.uses_packed_neon() {
        let packed = pack_decoder_gemm_left(left, params);
        for block in 0..params.column_blocks() {
            let values = decoder_gemm_column_block(&packed, right, bias, params, block);
            let first_column = block * COLUMNS_PER_BLOCK;
            for column in 0..values.columns() {
                for row in 0..ROWS_PER_BLOCK {
                    output[row * params.columns + first_column + column] =
                        values.value(column, row);
                }
            }
        }
        return Ok(());
    }
    for (linear, output) in output.iter_mut().enumerate() {
        *output = decoder_gemm_element(left, right, bias, params, linear);
    }
    Ok(())
}
