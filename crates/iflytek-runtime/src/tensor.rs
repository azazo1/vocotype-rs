use anyhow::{Result, bail};
use half::f16;
use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};

#[derive(Clone, Debug)]
pub(crate) enum TensorData {
    F16 { shape: Vec<usize>, data: Vec<f16> },
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I32 { shape: Vec<usize>, data: Vec<i32> },
}

impl TensorData {
    pub(crate) fn zeros_for_input(session: &Session, name: &str) -> Result<Self> {
        let outlet = session
            .inputs()
            .iter()
            .find(|input| input.name() == name)
            .ok_or_else(|| anyhow::anyhow!("ONNX model is missing input {}", name))?;
        let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
            bail!("ONNX input {} is not a tensor", name)
        };
        let shape = fixed_shape(shape, name)?;
        let length = shape.iter().product();
        match ty {
            TensorElementType::Float16 => Ok(Self::F16 {
                shape,
                data: vec![f16::ZERO; length],
            }),
            TensorElementType::Float32 => Ok(Self::F32 {
                shape,
                data: vec![0.0; length],
            }),
            TensorElementType::Int32 => Ok(Self::I32 {
                shape,
                data: vec![0; length],
            }),
            _ => bail!("ONNX input {} uses unsupported tensor type {}", name, ty),
        }
    }

    pub(crate) fn from_value(value: &DynValue, name: &str) -> Result<Self> {
        let ValueType::Tensor { ty, .. } = value.dtype() else {
            bail!("ONNX output {} is not a tensor", name)
        };
        match ty {
            TensorElementType::Float16 => {
                let (shape, data) = value.try_extract_tensor::<f16>()?;
                Ok(Self::F16 {
                    shape: fixed_shape(shape, name)?,
                    data: data.to_vec(),
                })
            }
            TensorElementType::Float32 => {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                Ok(Self::F32 {
                    shape: fixed_shape(shape, name)?,
                    data: data.to_vec(),
                })
            }
            TensorElementType::Int32 => {
                let (shape, data) = value.try_extract_tensor::<i32>()?;
                Ok(Self::I32 {
                    shape: fixed_shape(shape, name)?,
                    data: data.to_vec(),
                })
            }
            _ => bail!("ONNX output {} uses unsupported tensor type {}", name, ty),
        }
    }

    pub(crate) fn into_value(self) -> Result<DynValue> {
        match self {
            Self::F16 { shape, data } => Ok(Tensor::from_array((shape, data))?.into_dyn()),
            Self::F32 { shape, data } => Ok(Tensor::from_array((shape, data))?.into_dyn()),
            Self::I32 { shape, data } => Ok(Tensor::from_array((shape, data))?.into_dyn()),
        }
    }

    pub(crate) fn gather_rows(&self, parents: &[usize], max_rows: usize) -> Result<Self> {
        match self {
            Self::F32 { shape, data } => {
                let row_width = row_width(shape, data.len(), max_rows)?;
                let mut gathered = vec![0.0; data.len()];
                for (target, parent) in parents.iter().copied().enumerate() {
                    if parent >= max_rows {
                        bail!("decoder state parent index is outside row count")
                    }
                    let source = parent * row_width;
                    let destination = target * row_width;
                    gathered[destination..destination + row_width]
                        .copy_from_slice(&data[source..source + row_width]);
                }
                Ok(Self::F32 {
                    shape: shape.clone(),
                    data: gathered,
                })
            }
            Self::F16 { shape, data } => {
                let row_width = row_width(shape, data.len(), max_rows)?;
                let mut gathered = vec![f16::ZERO; data.len()];
                for (target, parent) in parents.iter().copied().enumerate() {
                    if parent >= max_rows {
                        bail!("decoder state parent index is outside row count")
                    }
                    let source = parent * row_width;
                    let destination = target * row_width;
                    gathered[destination..destination + row_width]
                        .copy_from_slice(&data[source..source + row_width]);
                }
                Ok(Self::F16 {
                    shape: shape.clone(),
                    data: gathered,
                })
            }
            Self::I32 { .. } => bail!("decoder recurrent state must be floating point"),
        }
    }
}

pub(crate) fn output_f32(value: &DynValue, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        bail!("ONNX output {} is not a tensor", name)
    };
    match ty {
        TensorElementType::Float32 => {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            Ok((fixed_shape(shape, name)?, data.to_vec()))
        }
        TensorElementType::Float16 => {
            let (shape, data) = value.try_extract_tensor::<f16>()?;
            Ok((
                fixed_shape(shape, name)?,
                data.iter().map(|value| value.to_f32()).collect(),
            ))
        }
        _ => bail!("ONNX output {} must use f16 or f32", name),
    }
}

pub(crate) fn output_f16(value: &DynValue, name: &str) -> Result<(Vec<usize>, Vec<f16>)> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        bail!("ONNX output {} is not a tensor", name)
    };
    match ty {
        TensorElementType::Float16 => {
            let (shape, data) = value.try_extract_tensor::<f16>()?;
            Ok((fixed_shape(shape, name)?, data.to_vec()))
        }
        TensorElementType::Float32 => {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            Ok((
                fixed_shape(shape, name)?,
                data.iter().map(|value| f16::from_f32(*value)).collect(),
            ))
        }
        _ => bail!("ONNX output {} must use f16 or f32", name),
    }
}

pub(crate) fn output_i32_bits(value: &DynValue, name: &str) -> Result<Vec<i32>> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        bail!("ONNX output {} is not a tensor", name)
    };
    match ty {
        TensorElementType::Int32 => Ok(value.try_extract_tensor::<i32>()?.1.to_vec()),
        TensorElementType::Float32 => Ok(value
            .try_extract_tensor::<f32>()?
            .1
            .iter()
            .map(|value| value.to_bits() as i32)
            .collect()),
        _ => bail!("ONNX output {} must use i32 or f32 bit patterns", name),
    }
}

fn fixed_shape(shape: &[i64], name: &str) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                anyhow::anyhow!("ONNX tensor {} has a dynamic or negative shape", name)
            })
        })
        .collect()
}

fn row_width(shape: &[usize], length: usize, max_rows: usize) -> Result<usize> {
    if shape.first().copied() != Some(max_rows) || !length.is_multiple_of(max_rows) {
        bail!("decoder recurrent state has an unexpected batch shape")
    }
    Ok(length / max_rows)
}

#[cfg(test)]
mod tests {
    use super::TensorData;

    #[test]
    fn recurrent_state_gathers_parent_rows_and_clears_inactive_rows() {
        let state = TensorData::F32 {
            shape: vec![4, 2],
            data: vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        };
        let gathered = state.gather_rows(&[2, 0], 4).unwrap();
        let TensorData::F32 { shape, data } = gathered else {
            panic!("expected f32 state")
        };
        assert_eq!(shape, [4, 2]);
        assert_eq!(data, [20.0, 21.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn recurrent_state_rejects_parent_outside_batch() {
        let state = TensorData::F16 {
            shape: vec![2, 1],
            data: vec![half::f16::ZERO; 2],
        };
        assert!(state.gather_rows(&[2], 2).is_err());
    }
}
