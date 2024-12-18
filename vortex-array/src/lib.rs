#![feature(once_cell_try)]
#![feature(trusted_len)]
//! Vortex crate containing core logic for encoding and memory representation of [arrays](ArrayData).
//!
//! At the heart of Vortex are [arrays](ArrayData) and [encodings](crate::encoding::EncodingVTable).
//! Arrays are typed views of memory buffers that hold [scalars](vortex_scalar::Scalar). These
//! buffers can be held in a number of physical encodings to perform lightweight compression that
//! exploits the particular data distribution of the array's values.
//!
//! Every data type recognized by Vortex also has a canonical physical encoding format, which
//! arrays can be [canonicalized](Canonical) into for ease of access in compute functions.
//!

pub use canonical::*;
pub use children::*;
pub use context::*;
pub use data::*;
pub use metadata::*;
pub use paste;
use vortex_dtype::DType;

use crate::encoding::ArrayEncodingRef;
use crate::nbytes::ArrayNBytes;
use crate::stats::ArrayStatistics;
use crate::validity::ArrayValidity;

pub mod accessor;
pub mod aliases;
pub mod array;
pub mod arrow;
pub mod builders;
mod canonical;
mod children;
pub mod compress;
pub mod compute;
mod context;
mod data;
pub mod encoding;
pub mod iter;
mod macros;
mod metadata;
pub mod nbytes;
pub mod patches;
pub mod stats;
pub mod stream;
pub mod tree;
pub mod validity;
pub mod variants;
pub mod visitor;

pub mod flatbuffers {
    //! Re-exported autogenerated code from the core Vortex flatbuffer definitions.
    pub use vortex_flatbuffers::array::*;
}

/// A depth-first pre-order iterator over a ArrayData.
pub struct ArrayChildrenIterator {
    stack: Vec<ArrayData>,
}

impl ArrayChildrenIterator {
    pub fn new(array: ArrayData) -> Self {
        Self { stack: vec![array] }
    }
}

impl Iterator for ArrayChildrenIterator {
    type Item = ArrayData;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.stack.pop()?;
        for child in next.children().into_iter().rev() {
            self.stack.push(child);
        }
        Some(next)
    }
}

pub trait ToArrayData {
    fn to_array(&self) -> ArrayData;
}

/// Consume `self` and turn it into an [`ArrayData`] infallibly.
///
/// Implementation of this array should never fail.
pub trait IntoArrayData {
    fn into_array(self) -> ArrayData;
}

/// Collects together the behavior of an array.
pub trait ArrayTrait:
    AsRef<ArrayData>
    + ArrayEncodingRef
    + ArrayDType
    + ArrayLen
    + ArrayNBytes
    + IntoCanonical
    + ArrayValidity
    + ArrayStatistics
{
}

pub trait ArrayDType {
    // TODO(ngates): move into ArrayTrait?
    fn dtype(&self) -> &DType;
}

pub trait ArrayLen {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;
}
