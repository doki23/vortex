mod compute;

use std::fmt::Display;
use std::sync::Arc;

use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_dtype::{match_each_native_ptype, DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect, VortexResult};

use crate::array::{NullArray, PrimitiveArray};
use crate::compute::{scalar_at, slice};
use crate::encoding::ids;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{ListArrayTrait, PrimitiveArrayTrait, VariantsVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoCanonical,
};

impl_encoding!("vortex.list", ids::LIST, List);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMetadata {
    validity: ValidityMetadata,
    elements_len: usize,
    offset_ptype: PType,
}

impl Display for ListMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListMetadata")
    }
}

// A list is valid if the:
// - offsets start at a value in elements
// - offsets are sorted
// - the final offset points to an element in the elements list, pointing to zero
//   if elements are empty.
// - final_offset >= start_offset
// - The size of the validity is the size-1 of the offset array

impl ListArray {
    pub fn try_new(
        elements: ArrayData,
        offsets: ArrayData,
        validity: Validity,
    ) -> VortexResult<Self> {
        let nullability = validity.nullability();
        let list_len = offsets.len() - 1;
        let element_len = elements.len();

        let validity_metadata = validity.to_metadata(list_len)?;

        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!(
                "Expected offsets to be an non-nullable integer type, got {:?}",
                offsets.dtype()
            );
        }

        if offsets.is_empty() {
            vortex_bail!("Offsets must have at least one element, [0] for an empty list");
        }

        let offset_ptype = PType::try_from(offsets.dtype())?;

        let list_dtype = DType::List(Arc::new(elements.dtype().clone()), nullability);

        let mut children = vec![elements, offsets];
        if let Some(val) = validity.into_array() {
            children.push(val);
        }

        Self::try_from_parts(
            list_dtype,
            list_len,
            ListMetadata {
                validity: validity_metadata,
                elements_len: element_len,
                offset_ptype,
            },
            children.into(),
            StatsSet::default(),
        )
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(2, &Validity::DTYPE, self.len())
                .vortex_expect("ListArray: validity child")
        })
    }

    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    // TODO: merge logic with varbin
    pub fn offset_at(&self, index: usize) -> usize {
        PrimitiveArray::try_from(self.offsets())
            .ok()
            .map(|p| {
                match_each_native_ptype!(p.ptype(), |$P| {
                    p.maybe_null_slice::<$P>()[index].as_()
                })
            })
            .unwrap_or_else(|| {
                scalar_at(self.offsets(), index)
                    .unwrap_or_else(|err| {
                        vortex_panic!(err, "Failed to get offset at index: {}", index)
                    })
                    .as_ref()
                    .try_into()
                    .vortex_expect("Failed to convert offset to usize")
            })
    }

    // TODO: fetches the elements at index
    pub fn elements_at(&self, index: usize) -> VortexResult<ArrayData> {
        if index >= self.len() {
            vortex_bail!("Index out of bounds: index={} len={}", index, self.len());
        }
        if !self.is_valid(index) {
            return Ok(NullArray::new(1).into_array());
        }
        let start = self.offset_at(index);
        let end = self.offset_at(index + 1);
        slice(self.elements(), start, end)
    }

    // TODO: fetches the offsets of the array ignoring validity
    pub fn offsets(&self) -> ArrayData {
        // TODO: find cheap transform
        self.as_ref()
            .child(1, &self.metadata().offset_ptype.into(), self.len() + 1)
            .vortex_expect("array contains offsets")
    }

    // TODO: fetches the elements of the array ignoring validity
    pub fn elements(&self) -> ArrayData {
        let dtype = self
            .dtype()
            .as_list_element()
            .vortex_expect("must be list dtype");
        self.as_ref()
            .child(0, dtype, self.metadata().elements_len)
            .vortex_expect("array contains elements")
    }
}

impl VariantsVTable<ListArray> for ListEncoding {
    fn as_list_array<'a>(&self, array: &'a ListArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }
}

impl ArrayTrait for ListArray {}

impl VisitorVTable<ListArray> for ListEncoding {
    fn accept(&self, array: &ListArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &array.offsets())?;
        visitor.visit_child("elements", &array.elements())?;
        visitor.visit_validity(&array.validity())
    }
}

impl IntoCanonical for ListArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::List(self))
    }
}

impl StatisticsVTable<ListArray> for ListEncoding {
    fn compute_statistics(&self, _array: &ListArray, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl ListArrayTrait for ListArray {}

impl ValidityVTable<ListArray> for ListEncoding {
    fn is_valid(&self, array: &ListArray, index: usize) -> bool {
        array.is_valid(index)
    }

    fn logical_validity(&self, array: &ListArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_dtype::{Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::list::ListArray;
    use crate::array::PrimitiveArray;
    use crate::compute::scalar_at;
    use crate::validity::Validity;
    use crate::{ArrayLen, IntoArrayData};

    #[test]
    fn test_empty_list_array() {
        let elements = PrimitiveArray::from(vec![] as Vec<u32>);
        let offsets = PrimitiveArray::from(vec![0]);
        let validity = Validity::AllValid;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(0, list.len());
    }

    #[test]
    fn test_simple_list_array() {
        let elements = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);
        let offsets = PrimitiveArray::from(vec![0, 2, 4, 5]);
        let validity = Validity::AllValid;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(
            Scalar::list(
                Arc::new(PType::I32.into()),
                vec![1.into(), 2.into()],
                Nullability::Nullable
            ),
            scalar_at(&list, 0).unwrap()
        );
        assert_eq!(
            Scalar::list(
                Arc::new(PType::I32.into()),
                vec![3.into(), 4.into()],
                Nullability::Nullable
            ),
            scalar_at(&list, 1).unwrap()
        );
        assert_eq!(
            Scalar::list(
                Arc::new(PType::I32.into()),
                vec![5.into()],
                Nullability::Nullable
            ),
            scalar_at(&list, 2).unwrap()
        );
    }
}
