use vortex_array::compute::ScalarAtFn;
use vortex_array::ArrayDType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{unpack_single, BitPackedArray, BitPackedEncoding};

impl ScalarAtFn<BitPackedArray> for BitPackedEncoding {
    fn scalar_at(&self, array: &BitPackedArray, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return Ok(patch);
            }
        }
        unpack_single(array, index)?.cast(array.dtype())
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::scalar_at;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_array::IntoArrayData;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::BitPackedArray;

    #[test]
    fn invalid_patches() {
        // SAFETY: using unsigned PType
        let packed_array = unsafe {
            BitPackedArray::new_unchecked(
                Buffer::from(vec![0u8; 128]),
                PType::U32,
                Validity::AllInvalid,
                Some(Patches::new(
                    8,
                    PrimitiveArray::from_vec(vec![1u32], NonNullable).into_array(),
                    PrimitiveArray::from_vec(vec![999u32], Validity::AllValid).into_array(),
                )),
                1,
                8,
            )
        }
        .unwrap()
        .into_array();
        assert_eq!(
            scalar_at(&packed_array, 1).unwrap(),
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable))
        );
    }

    #[test]
    fn test_scalar_at() {
        let values = (0u32..257).collect_vec();
        let uncompressed = PrimitiveArray::from(values.clone()).into_array();
        let packed = BitPackedArray::encode(&uncompressed, 8).unwrap();
        assert!(packed.patches().is_some());

        let patches = packed.patches().unwrap().indices().clone();
        assert_eq!(
            usize::try_from(&scalar_at(patches, 0).unwrap()).unwrap(),
            256
        );

        values.iter().enumerate().for_each(|(i, v)| {
            assert_eq!(
                u32::try_from(scalar_at(packed.as_ref(), i).unwrap().as_ref()).unwrap(),
                *v
            );
        });
    }
}
