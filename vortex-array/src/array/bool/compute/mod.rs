use crate::array::{BoolArray, BoolEncoding};
use crate::compute::unary::{FillForwardFn, ScalarAtFn};
use crate::compute::{AndFn, ArrayCompute, ComputeVTable, FilterFn, OrFn, SliceFn, TakeFn};
use crate::ArrayData;

mod boolean;

mod fill;
pub mod filter;
mod flatten;
mod scalar_at;
mod slice;
mod take;

impl ArrayCompute for BoolArray {
    fn and(&self) -> Option<&dyn AndFn> {
        Some(self)
    }

    fn or(&self) -> Option<&dyn OrFn> {
        Some(self)
    }
}

impl ComputeVTable for BoolEncoding {
    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}
