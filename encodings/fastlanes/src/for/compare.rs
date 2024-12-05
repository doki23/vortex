use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{decompress, FoRArray, FoREncoding};

impl CompareFn<FoRArray> for FoREncoding {
    fn compare(
        &self,
        lhs: &FoRArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // this is cheap
        let owned_lhs = lhs.clone();
        let decompressed_lhs = decompress(owned_lhs)?;
        compare(decompressed_lhs, rhs, operator).map(|array_data| Some(array_data))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{compare, Operator};
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;

    use crate::for_compress;

    #[test]
    fn test_for_compare() {
        let lhs = PrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5], Validity::AllValid);
        let lhs = for_compress(&lhs).unwrap();
        let rhs = PrimitiveArray::from_vec(vec![1i32, 2, 9, 4, 5], Validity::AllValid);
        assert_eq!(
            compare(lhs, rhs, Operator::Eq)
                .unwrap()
                .into_bool()
                .unwrap()
                .boolean_buffer(),
            vec![true, true, false, true, true].into()
        );
    }
}
