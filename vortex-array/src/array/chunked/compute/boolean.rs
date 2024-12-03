use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{
    binary_boolean, slice, BinaryBooleanFn, BinaryOperator,
};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl BinaryBooleanFn<ChunkedArray> for ChunkedEncoding {
    fn binary_boolean(
        &self,
        lhs: &ChunkedArray,
        rhs: &ArrayData,
        op: BinaryOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let mut idx = 0;
        let mut chunks = Vec::with_capacity(lhs.nchunks());
        let mut nullability = Nullability::Nullable;

        for chunk in lhs.chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let result = binary_boolean(&chunk, &sliced, op)?;
            nullability = result.dtype().nullability();
            chunks.push(result);
            idx += chunk.len();
        }

        Ok(Some(
            ChunkedArray::try_new(chunks, DType::Bool(nullability))?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::array::{BoolArray, ChunkedArray};
    use crate::compute::{binary_boolean, BinaryOperator};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_bin_bool_chunked() {
        let arr0 = BoolArray::from_iter(vec![true, false]).into_array();
        let arr1 = BoolArray::from_iter(vec![false, false, true]).into_array();
        let chunked1 =
            ChunkedArray::try_new(vec![arr0, arr1], DType::Bool(Nullability::NonNullable)).unwrap();

        let arr2 = BoolArray::from_iter(vec![false, true]).into_array();
        let arr3 = BoolArray::from_iter(vec![false, false, false]).into_array();
        let chunked2 =
            ChunkedArray::try_new(vec![arr2, arr3], DType::Bool(Nullability::NonNullable)).unwrap();

        assert_eq!(
            binary_boolean(
                &chunked1.into_array(),
                &chunked2.into_array(),
                BinaryOperator::Or
            )
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer(),
            vec![true, true, false, false, true].into()
        );
    }
}
