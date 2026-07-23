use half::f16;
use iflytek_core::{
    CUSTOM_OP_DOMAIN, Conv2dParams, decoder_cos, decoder_gemm_column_block, decoder_gemm_element,
    decoder_gemm_params, decoder_layer_norm, decoder_log_softmax, decoder_matmul,
    decoder_reduce_sum, decoder_sigmoid, decoder_sin,
    depthwise_conv_element, depthwise_conv_output_len, gelu_f16, gemm_f16_element,
    gemm_f16_into, gemm_f16_output_len, gemm_f16_uses_accelerate,
    gemm_f16_uses_packed_neon, layer_norm_f16, matmul_f16, original_add,
    original_multiply, pack_decoder_gemm_left_into, sigmoid_f16, softmax_f16, standard_conv_element,
    standard_conv_into, standard_conv_output_len, standard_conv_uses_accelerate,
};
use ort::operator::{Kernel, KernelAttributes, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput};
use ort::value::TensorElementType;
use ort::{Error, Result};

fn error(message: impl Into<String>) -> Error {
    Error::new(message.into())
}

fn shape_elements(shape: &ort::value::Shape) -> usize {
    shape.iter().copied().map(|value| value as usize).product()
}

#[derive(Clone, Copy)]
struct DisjointOutput<T> {
    pointer: *mut T,
    length: usize,
}

impl<T> DisjointOutput<T> {
    fn new(values: &mut [T]) -> Self {
        Self {
            pointer: values.as_mut_ptr(),
            length: values.len(),
        }
    }

    fn write(self, index: usize, value: T) {
        assert!(index < self.length);
        unsafe { self.pointer.add(index).write(value) };
    }
}

unsafe impl<T: Send> Send for DisjointOutput<T> {}
unsafe impl<T: Send> Sync for DisjointOutput<T> {}

struct F16Unary {
    name: &'static str,
    operation: fn(&[f16]) -> Vec<f16>,
}

impl Operator for F16Unary {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![OperatorInput::required(TensorElementType::Float16)]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let operation = self.operation;
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing unary input"))?;
            let (shape, values) = input.try_extract_tensor::<f16>()?;
            let computed = operation(values);
            let mut output = context
                .output(0, shape.to_vec())?
                .ok_or_else(|| error("missing unary output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F32Unary {
    name: &'static str,
    operation: fn(&[f32]) -> Vec<f32>,
}

impl Operator for F32Unary {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![OperatorInput::required(TensorElementType::Float32)]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let operation = self.operation;
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing unary input"))?;
            let (shape, values) = input.try_extract_tensor::<f32>()?;
            let computed = operation(values);
            let mut output = context
                .output(0, shape.to_vec())?
                .ok_or_else(|| error("missing unary output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f32>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

type F16BinaryOperation = fn(
    &[usize],
    &[f16],
    &[usize],
    &[f16],
) -> anyhow::Result<(Vec<usize>, Vec<f16>)>;

struct F16Binary {
    name: &'static str,
    operation: F16BinaryOperation,
}

impl Operator for F16Binary {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let operation = self.operation;
        Ok(Box::new(move |context: &KernelContext| {
            let left = context.input(0)?.ok_or_else(|| error("missing left input"))?;
            let right = context.input(1)?.ok_or_else(|| error("missing right input"))?;
            let (left_shape, left_values) = left.try_extract_tensor::<f16>()?;
            let (right_shape, right_values) = right.try_extract_tensor::<f16>()?;
            let left_shape = shape_usizes(left_shape)?;
            let right_shape = shape_usizes(right_shape)?;
            let (output_shape, computed) = operation(
                &left_shape,
                left_values,
                &right_shape,
                right_values,
            )
            .map_err(|e| error(e.to_string()))?;
            let mut output = context
                .output(0, output_shape)?
                .ok_or_else(|| error("missing binary output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F16Gemm {
    name: &'static str,
}

impl Operator for F16Gemm {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(move |context: &KernelContext| {
            let left = context.input(0)?.ok_or_else(|| error("missing GEMM left input"))?;
            let right = context.input(1)?.ok_or_else(|| error("missing GEMM right input"))?;
            let bias = context.input(2)?.ok_or_else(|| error("missing GEMM bias input"))?;
            let (left_shape, left_values) = left.try_extract_tensor::<f16>()?;
            let (right_shape, right_values) = right.try_extract_tensor::<f16>()?;
            let (_, bias_values) = bias.try_extract_tensor::<f16>()?;
            if left_shape.len() != 2 || right_shape.len() != 2 {
                return Err(error("GEMM expects rank-two inputs"));
            }
            let rows = left_shape[0] as usize;
            let inner = left_shape[1] as usize;
            let columns = right_shape[0] as usize;
            let output_len = gemm_f16_output_len(
                left_values,
                right_values,
                bias_values,
                rows,
                inner,
                columns,
            )
            .map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, vec![rows, columns])?.ok_or_else(|| error("missing GEMM output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            if target.len() != output_len {
                return Err(error("GEMM output allocation has an invalid size"));
            }
            if gemm_f16_uses_accelerate(rows) || gemm_f16_uses_packed_neon(rows) {
                gemm_f16_into(
                    left_values,
                    right_values,
                    bias_values,
                    rows,
                    inner,
                    columns,
                    target,
                )
                .map_err(|e| error(e.to_string()))?;
            } else {
                let target = DisjointOutput::new(target);
                context.par_for(output_len, 0, move |linear| {
                    target.write(
                        linear,
                        gemm_f16_element(
                            left_values,
                            right_values,
                            bias_values,
                            inner,
                            columns,
                            linear,
                        ),
                    );
                })?;
            }
            Ok(())
        }))
    }
}

struct DecoderF16Gemm {
    name: &'static str,
}

impl Operator for DecoderF16Gemm {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let packed_left_workspace = std::sync::Mutex::new(Vec::<f16>::new());
        Ok(Box::new(move |context: &KernelContext| {
            let left = context.input(0)?.ok_or_else(|| error("missing decoder GEMM left input"))?;
            let right = context.input(1)?.ok_or_else(|| error("missing decoder GEMM right input"))?;
            let bias = context.input(2)?.ok_or_else(|| error("missing decoder GEMM bias input"))?;
            let (left_shape, left_values) = left.try_extract_tensor::<f16>()?;
            let (right_shape, right_values) = right.try_extract_tensor::<f16>()?;
            let (_, bias_values) = bias.try_extract_tensor::<f16>()?;
            if left_shape.len() != 2 || right_shape.len() != 2 {
                return Err(error("decoder GEMM expects rank-two inputs"));
            }
            let rows = left_shape[0] as usize;
            let inner = left_shape[1] as usize;
            let columns = right_shape[0] as usize;
            let params = decoder_gemm_params(
                left_values,
                right_values,
                bias_values,
                rows,
                inner,
                columns,
            )
            .map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, vec![rows, columns])?.ok_or_else(|| error("missing decoder GEMM output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            if target.len() != params.output_len() {
                return Err(error("decoder GEMM output allocation has an invalid size"));
            }
            let target = DisjointOutput::new(target);
            if params.uses_packed_neon() {
                let mut packed_left = packed_left_workspace
                    .lock()
                    .map_err(|_| error("decoder GEMM workspace lock poisoned"))?;
                pack_decoder_gemm_left_into(left_values, params, &mut packed_left);
                let packed_left = packed_left.as_slice();
                context.par_for(params.column_blocks(), 0, move |block| {
                    let values = decoder_gemm_column_block(
                        packed_left,
                        right_values,
                        bias_values,
                        params,
                        block,
                    );
                    let first_column = block * 4;
                    for column in 0..values.columns() {
                        for row in 0..params.rows() {
                            target.write(
                                row * params.columns() + first_column + column,
                                values.value(column, row),
                            );
                        }
                    }
                })?;
            } else {
                context.par_for(params.output_len(), 0, move |linear| {
                    target.write(
                        linear,
                        decoder_gemm_element(
                            left_values,
                            right_values,
                            bias_values,
                            params,
                            linear,
                        ),
                    );
                })?;
            }
            Ok(())
        }))
    }
}

struct F16MatMul {
    name: &'static str,
}

impl Operator for F16MatMul {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(move |context: &KernelContext| {
            let left = context.input(0)?.ok_or_else(|| error("missing MatMul left input"))?;
            let right = context.input(1)?.ok_or_else(|| error("missing MatMul right input"))?;
            let (left_shape, left_values) = left.try_extract_tensor::<f16>()?;
            let (right_shape, right_values) = right.try_extract_tensor::<f16>()?;
            let (output_shape, computed) = batched_matmul_f16(
                left_shape,
                left_values,
                right_shape,
                right_values,
            )?;
            let mut output = context.output(0, output_shape)?.ok_or_else(|| error("missing MatMul output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F32MatMul {
    name: &'static str,
}

impl Operator for F32MatMul {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(move |context: &KernelContext| {
            let left = context.input(0)?.ok_or_else(|| error("missing decoder MatMul left input"))?;
            let right = context.input(1)?.ok_or_else(|| error("missing decoder MatMul right input"))?;
            let (left_shape, left_values) = left.try_extract_tensor::<f32>()?;
            let (right_shape, right_values) = right.try_extract_tensor::<f32>()?;
            let (output_shape, computed) = batched_matmul_f32(
                left_shape,
                left_values,
                right_shape,
                right_values,
            )?;
            let mut output = context.output(0, output_shape)?.ok_or_else(|| error("missing decoder MatMul output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f32>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F16Softmax;

impl Operator for F16Softmax {
    fn name(&self) -> &str {
        "OriginalSoftmax"
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![OperatorInput::required(TensorElementType::Float16)]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(|context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing softmax input"))?;
            let (shape, values) = input.try_extract_tensor::<f16>()?;
            let width = usize::try_from(
                *shape.last().ok_or_else(|| error("softmax input has no width"))?,
            )
            .map_err(|_| error("softmax width is negative"))?;
            if width == 0 {
                return Err(error("softmax width must be positive"));
            }
            let rows = shape_elements(shape) / width;
            let computed = softmax_f16(values, rows, width).map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, shape.to_vec())?.ok_or_else(|| error("missing softmax output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F32LogSoftmax;

impl Operator for F32LogSoftmax {
    fn name(&self) -> &str {
        "DecoderLogSoftmax"
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![OperatorInput::required(TensorElementType::Float32)]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(|context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing log-softmax input"))?;
            let (shape, values) = input.try_extract_tensor::<f32>()?;
            if shape.len() != 2 {
                return Err(error("decoder log-softmax expects a rank-two tensor"));
            }
            let width = usize::try_from(shape[1])
                .map_err(|_| error("decoder log-softmax width is negative"))?;
            if width == 0 {
                return Err(error("decoder log-softmax width must be positive"));
            }
            let rows = shape_elements(shape) / width;
            let computed = decoder_log_softmax(values, rows, width).map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, shape.to_vec())?.ok_or_else(|| error("missing log-softmax output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f32>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F16LayerNorm {
    name: &'static str,
}

impl Operator for F16LayerNorm {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let epsilon = attributes.get::<f32>("epsilon").unwrap_or(1.0e-5);
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing layer norm input"))?;
            let scale = context.input(1)?.ok_or_else(|| error("missing layer norm scale"))?;
            let bias = context.input(2)?.ok_or_else(|| error("missing layer norm bias"))?;
            let (shape, input_values) = input.try_extract_tensor::<f16>()?;
            let (_, scale_values) = scale.try_extract_tensor::<f16>()?;
            let (_, bias_values) = bias.try_extract_tensor::<f16>()?;
            let width = usize::try_from(
                *shape.last().ok_or_else(|| error("layer norm input has no width"))?,
            )
            .map_err(|_| error("layer norm width is negative"))?;
            if width == 0 {
                return Err(error("layer norm width must be positive"));
            }
            let rows = shape_elements(shape) / width;
            let computed = layer_norm_f16(input_values, scale_values, bias_values, rows, width, epsilon)
                .map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, shape.to_vec())?.ok_or_else(|| error("missing layer norm output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F32LayerNorm {
    name: &'static str,
}

impl Operator for F32LayerNorm {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let epsilon = attributes.get::<f32>("epsilon").unwrap_or(1.0e-5);
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing decoder layer norm input"))?;
            let scale = context.input(1)?.ok_or_else(|| error("missing decoder layer norm scale"))?;
            let bias = context.input(2)?.ok_or_else(|| error("missing decoder layer norm bias"))?;
            let (shape, input_values) = input.try_extract_tensor::<f32>()?;
            let (_, scale_values) = scale.try_extract_tensor::<f32>()?;
            let (_, bias_values) = bias.try_extract_tensor::<f32>()?;
            let width = usize::try_from(
                *shape
                    .last()
                    .ok_or_else(|| error("decoder layer norm input has no width"))?,
            )
            .map_err(|_| error("decoder layer norm width is negative"))?;
            if width == 0 {
                return Err(error("decoder layer norm width must be positive"));
            }
            let rows = shape_elements(shape) / width;
            let computed = decoder_layer_norm(input_values, scale_values, bias_values, rows, width, epsilon)
                .map_err(|e| error(e.to_string()))?;
            let mut output = context.output(0, shape.to_vec())?.ok_or_else(|| error("missing decoder layer norm output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f32>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct F16Conv {
    name: &'static str,
    depthwise: bool,
}

impl Operator for F16Conv {
    fn name(&self) -> &str {
        self.name
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Float16),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let depthwise = self.depthwise;
        let stride_height = attributes.get::<i64>("stride_height").unwrap_or(1) as usize;
        let stride_width = attributes.get::<i64>("stride_width").unwrap_or(1) as usize;
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing convolution input"))?;
            let weights = context.input(1)?.ok_or_else(|| error("missing convolution weights"))?;
            let bias = context.input(2)?.ok_or_else(|| error("missing convolution bias"))?;
            let (input_shape, input_values) = input.try_extract_tensor::<f16>()?;
            let (weight_shape, weight_values) = weights.try_extract_tensor::<f16>()?;
            let (_, bias_values) = bias.try_extract_tensor::<f16>()?;
            if input_shape.len() != 4 || weight_shape.len() != 4 {
                return Err(error("convolution expects rank-four inputs"));
            }
            let batch = input_shape[0] as usize;
            let input_channels = input_shape[1] as usize;
            let input_height = input_shape[2] as usize;
            let input_width = input_shape[3] as usize;
            let output_channels = weight_shape[0] as usize;
            let kernel_height = weight_shape[2] as usize;
            let kernel_width = weight_shape[3] as usize;
            let params = Conv2dParams {
                batch,
                input_channels,
                input_height,
                input_width,
                output_channels: if depthwise { input_channels } else { output_channels },
                kernel_height,
                kernel_width,
                stride_height: if depthwise { 1 } else { stride_height },
                stride_width: if depthwise { 1 } else { stride_width },
            };
            let output_len = if depthwise {
                depthwise_conv_output_len(input_values, weight_values, bias_values, params)
            } else {
                standard_conv_output_len(input_values, weight_values, bias_values, params)
            }
            .map_err(|e| error(e.to_string()))?;
            let output_height = params.output_height();
            let output_width = params.output_width();
            let output_channels = params.output_channels;
            let mut output = context.output(0, vec![batch, output_channels, output_height, output_width])?.ok_or_else(|| error("missing convolution output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f16>()?;
            if target.len() != output_len {
                return Err(error("convolution output allocation has an invalid size"));
            }
            if !depthwise && standard_conv_uses_accelerate(params) {
                standard_conv_into(input_values, weight_values, bias_values, params, target)
                    .map_err(|e| error(e.to_string()))?;
            } else {
                let target = DisjointOutput::new(target);
                context.par_for(output_len, 0, move |linear| {
                    let value = if depthwise {
                        depthwise_conv_element(
                            input_values,
                            weight_values,
                            bias_values,
                            params,
                            linear,
                        )
                    } else {
                        standard_conv_element(
                            input_values,
                            weight_values,
                            bias_values,
                            params,
                            linear,
                        )
                    };
                    target.write(linear, value);
                })?;
            }
            Ok(())
        }))
    }
}

struct F32ReduceSum;

impl Operator for F32ReduceSum {
    fn name(&self) -> &str {
        "DecoderReduceSum"
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Int64),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let keepdims = attributes.get::<i64>("keepdims").unwrap_or(1);
        if !matches!(keepdims, 0 | 1) {
            return Err(error("DecoderReduceSum keepdims must be zero or one"));
        }
        Ok(Box::new(move |context: &KernelContext| {
            let input = context.input(0)?.ok_or_else(|| error("missing reduce input"))?;
            let axes = context.input(1)?.ok_or_else(|| error("missing reduce axes"))?;
            let (shape, input_values) = input.try_extract_tensor::<f32>()?;
            let (_, axes_values) = axes.try_extract_tensor::<i64>()?;
            if axes_values != [-1] {
                return Err(error("DecoderReduceSum only supports axis -1"));
            }
            let width = usize::try_from(
                *shape.last().ok_or_else(|| error("reduce input has no width"))?,
            )
            .map_err(|_| error("reduce width is negative"))?;
            if width == 0 {
                return Err(error("reduce width must be positive"));
            }
            let rows = shape_elements(shape) / width;
            let computed = decoder_reduce_sum(input_values, rows, width).map_err(|e| error(e.to_string()))?;
            let mut output_shape = shape.to_vec();
            if keepdims == 0 {
                output_shape.pop();
            } else {
                *output_shape.last_mut().ok_or_else(|| error("reduce output has no width"))? = 1;
            }
            let mut output = context.output(0, output_shape)?.ok_or_else(|| error("missing reduce output"))?;
            let (_, target) = output.try_extract_tensor_mut::<f32>()?;
            target.copy_from_slice(&computed);
            Ok(())
        }))
    }
}

struct MatMulLayout {
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    batch_shape: Vec<usize>,
    output_shape: Vec<usize>,
    rows: usize,
    inner: usize,
    columns: usize,
}

fn matmul_layout(
    left: &ort::value::Shape,
    right: &ort::value::Shape,
) -> Result<MatMulLayout> {
    if left.len() < 2 || right.len() < 2 {
        return Err(error("MatMul inputs must have rank at least two"));
    }
    let rank = left.len().max(right.len());
    let mut left_shape = vec![1; rank - left.len()];
    left_shape.extend(left.iter().map(|value| *value as usize));
    let mut right_shape = vec![1; rank - right.len()];
    right_shape.extend(right.iter().map(|value| *value as usize));
    let rows = left_shape[rank - 2];
    let inner = left_shape[rank - 1];
    let columns = right_shape[rank - 1];
    if rows == 0 || inner == 0 || columns == 0 || right_shape[rank - 2] != inner {
        return Err(error("MatMul matrix dimensions are incompatible"));
    }
    let mut batch_shape = Vec::with_capacity(rank - 2);
    for axis in 0..rank - 2 {
        let left_dimension = left_shape[axis];
        let right_dimension = right_shape[axis];
        if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
            return Err(error("MatMul batch dimensions do not broadcast"));
        }
        batch_shape.push(left_dimension.max(right_dimension));
    }
    let mut output_shape = batch_shape.clone();
    output_shape.push(rows);
    output_shape.push(columns);
    Ok(MatMulLayout {
        left_shape,
        right_shape,
        batch_shape,
        output_shape,
        rows,
        inner,
        columns,
    })
}

fn shape_usizes(shape: &ort::value::Shape) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| error("runtime tensor shape contains a negative dimension"))
        })
        .collect()
}

fn batch_matrix_offset(
    mut batch: usize,
    source_shape: &[usize],
    output_batch_shape: &[usize],
) -> usize {
    let mut offset = 0;
    for axis in (0..output_batch_shape.len()).rev() {
        let coordinate = batch % output_batch_shape[axis];
        batch /= output_batch_shape[axis];
        if source_shape[axis] != 1 {
            let stride = source_shape[axis + 1..].iter().product::<usize>();
            offset += coordinate * stride;
        }
    }
    offset
}

fn batched_matmul_f16(
    left_shape: &ort::value::Shape,
    left: &[f16],
    right_shape: &ort::value::Shape,
    right: &[f16],
) -> Result<(Vec<usize>, Vec<f16>)> {
    let layout = matmul_layout(left_shape, right_shape)?;
    let batches = layout.batch_shape.iter().product::<usize>().max(1);
    let mut output = Vec::with_capacity(batches * layout.rows * layout.columns);
    for batch in 0..batches {
        let left_offset = batch_matrix_offset(batch, &layout.left_shape, &layout.batch_shape);
        let right_offset = batch_matrix_offset(batch, &layout.right_shape, &layout.batch_shape);
        let left_end = left_offset + layout.rows * layout.inner;
        let right_end = right_offset + layout.inner * layout.columns;
        let values = matmul_f16(
            left.get(left_offset..left_end).ok_or_else(|| error("MatMul left batch is out of bounds"))?,
            right.get(right_offset..right_end).ok_or_else(|| error("MatMul right batch is out of bounds"))?,
            layout.rows,
            layout.inner,
            layout.columns,
        )
        .map_err(|e| error(e.to_string()))?;
        output.extend(values);
    }
    Ok((layout.output_shape, output))
}

fn batched_matmul_f32(
    left_shape: &ort::value::Shape,
    left: &[f32],
    right_shape: &ort::value::Shape,
    right: &[f32],
) -> Result<(Vec<usize>, Vec<f32>)> {
    let layout = matmul_layout(left_shape, right_shape)?;
    let batches = layout.batch_shape.iter().product::<usize>().max(1);
    let mut output = Vec::with_capacity(batches * layout.rows * layout.columns);
    for batch in 0..batches {
        let left_offset = batch_matrix_offset(batch, &layout.left_shape, &layout.batch_shape);
        let right_offset = batch_matrix_offset(batch, &layout.right_shape, &layout.batch_shape);
        let left_end = left_offset + layout.rows * layout.inner;
        let right_end = right_offset + layout.inner * layout.columns;
        let values = decoder_matmul(
            left.get(left_offset..left_end).ok_or_else(|| error("decoder MatMul left batch is out of bounds"))?,
            right.get(right_offset..right_end).ok_or_else(|| error("decoder MatMul right batch is out of bounds"))?,
            layout.rows,
            layout.inner,
            layout.columns,
        )
        .map_err(|e| error(e.to_string()))?;
        output.extend(values);
    }
    Ok((layout.output_shape, output))
}

pub fn operator_domain() -> Result<OperatorDomain> {
    let domain = OperatorDomain::new(CUSTOM_OP_DOMAIN)?;
    let domain = domain.add(F16Gemm { name: "OriginalGemm" })?;
    let domain = domain.add(F16MatMul { name: "OriginalMatMul" })?;
    let domain = domain.add(F16Conv { name: "OriginalConv", depthwise: false })?;
    let domain = domain.add(F16Conv { name: "OriginalDepthwiseConv", depthwise: true })?;
    let domain = domain.add(F16Binary { name: "OriginalAdd", operation: original_add })?;
    let domain = domain.add(F16Binary { name: "OriginalMultiply", operation: original_multiply })?;
    let domain = domain.add(F16LayerNorm { name: "OriginalLayerNormalization" })?;
    let domain = domain.add(F16Unary { name: "OriginalSigmoid", operation: sigmoid_f16 })?;
    let domain = domain.add(F16Softmax)?;
    let domain = domain.add(F16Unary { name: "OriginalGelu", operation: gelu_f16 })?;
    let domain = domain.add(DecoderF16Gemm { name: "DecoderGemm" })?;
    let domain = domain.add(F32MatMul { name: "DecoderMatMul" })?;
    let domain = domain.add(F32LayerNorm { name: "DecoderLayerNormalization" })?;
    let domain = domain.add(F32Unary { name: "DecoderSigmoid", operation: decoder_sigmoid })?;
    let domain = domain.add(F32Unary { name: "DecoderCos", operation: decoder_cos })?;
    let domain = domain.add(F32Unary { name: "DecoderSin", operation: decoder_sin })?;
    let domain = domain.add(F32ReduceSum)?;
    let domain = domain.add(F32LogSoftmax)?;
    Ok(domain)
}
