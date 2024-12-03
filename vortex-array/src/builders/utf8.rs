use std::any::Any;

use arrow_array::builder::{ArrayBuilder as _, StringViewBuilder};
use arrow_array::Array;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::arrow::FromArrowArray;
use crate::builders::ArrayBuilder;
use crate::ArrayData;

pub struct Utf8Builder {
    inner: StringViewBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl Utf8Builder {
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: StringViewBuilder::with_capacity(capacity),
            nullability,
            dtype: DType::Utf8(nullability),
        }
    }

    pub fn append_value<S: AsRef<str>>(&mut self, value: S) {
        self.inner.append_value(value.as_ref())
    }

    pub fn append_option<S: AsRef<str>>(&mut self, value: Option<S>) {
        self.inner.append_option(value.as_ref())
    }
}

impl ArrayBuilder for Utf8Builder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.inner.append_value("")
        }
    }

    fn append_nulls(&mut self, n: usize) {
        for _ in 0..n {
            self.inner.append_null()
        }
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let arrow = self.inner.finish();

        if !self.dtype().is_nullable() && arrow.null_count() > 0 {
            vortex_bail!("Non-nullable builder has null values");
        }

        Ok(ArrayData::from_arrow(&arrow, self.nullability.into()))
    }
}
