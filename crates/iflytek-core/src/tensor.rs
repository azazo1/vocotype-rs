use std::fmt;

#[derive(Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &self.data)
            .finish()
    }
}

impl<T> Tensor<T> {
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<T>) -> anyhow::Result<Self> {
        let shape = shape.into();
        let expected = shape.iter().try_fold(1usize, |value, dimension| {
            value.checked_mul(*dimension)
        });
        if expected != Some(data.len()) {
            anyhow::bail!("tensor shape does not match data length")
        }
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_parts(self) -> (Vec<usize>, Vec<T>) {
        (self.shape, self.data)
    }
}

impl<T: Clone> Tensor<T> {
    pub fn filled(shape: impl Into<Vec<usize>>, value: T) -> anyhow::Result<Self> {
        let shape = shape.into();
        let length = shape.iter().try_fold(1usize, |value, dimension| {
            value.checked_mul(*dimension)
        });
        let Some(length) = length else {
            anyhow::bail!("tensor shape overflows element count")
        };
        Ok(Self {
            shape,
            data: vec![value; length],
        })
    }
}
