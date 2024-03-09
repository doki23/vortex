use crate::compute::as_contiguous::AsContiguousFn;
use crate::compute::search_sorted::SearchSortedFn;
use cast::{CastBoolFn, CastPrimitiveFn};
use fill::FillForwardFn;
use patch::PatchFn;
use scalar_at::ScalarAtFn;
use take::TakeFn;

pub mod add;
pub mod as_contiguous;
pub mod cast;
pub mod fill;
pub mod patch;
pub mod repeat;
pub mod scalar_at;
pub mod search_sorted;
pub mod take;

pub trait ArrayCompute {
    fn as_contiguous(&self) -> Option<&dyn AsContiguousFn> {
        None
    }

    fn cast_bool(&self) -> Option<&dyn CastBoolFn> {
        None
    }

    fn cast_primitive(&self) -> Option<&dyn CastPrimitiveFn> {
        None
    }

    fn fill_forward(&self) -> Option<&dyn FillForwardFn> {
        None
    }

    fn patch(&self) -> Option<&dyn PatchFn> {
        None
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        None
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        None
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        None
    }
}