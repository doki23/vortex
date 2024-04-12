use vortex::match_each_native_ptype;
use vortex::ptype::{NativePType, PType};
use vortex_error::{vortex_err, VortexResult};
use vortex_schema::DType;

use crate::array::primitive::PrimitiveArray;
use crate::compute::cast::CastFn;
use crate::{IntoArray, OwnedArray, ToArrayData};

impl CastFn for PrimitiveArray<'_> {
    fn cast(&self, dtype: &DType) -> VortexResult<OwnedArray> {
        // TODO(ngates): check validity
        let ptype = PType::try_from(dtype)?;
        if ptype == self.ptype() {
            return Ok(self.to_array_data().into_array());
        }

        match_each_native_ptype!(ptype, |$T| {
            Ok(PrimitiveArray::from_vec(
                cast::<$T>(self)?,
                self.validity().clone(),
            ).into_array())
        })
    }
}

fn cast<T: NativePType>(array: &PrimitiveArray) -> VortexResult<Vec<T>> {
    match_each_native_ptype!(array.ptype(), |$E| {
        array
            .typed_data::<$E>()
            .iter()
            // TODO(ngates): allow configurable checked/unchecked casting
            .map(|&v| {
                T::from(v).ok_or_else(|| {
                    vortex_err!(ComputeError: "Failed to cast {} to {:?}", v, T::PTYPE)
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod test {
    use vortex::ptype::PType;
    use vortex_error::VortexError;

    use crate::array::primitive::PrimitiveArray;
    use crate::{compute, IntoArray};

    #[test]
    fn cast_u32_u8() {
        let arr = vec![0u32, 10, 200].into_array();
        let p =
            PrimitiveArray::try_from(compute::cast::cast(&arr, PType::U8.into()).unwrap()).unwrap();
        assert_eq!(p.typed_data::<u8>(), vec![0u8, 10, 200]);
    }

    #[test]
    fn cast_u32_f32() {
        let arr = vec![0u32, 10, 200].into_array();
        let u8arr = PrimitiveArray::try_from(compute::cast::cast(&arr, PType::F32.into()).unwrap())
            .unwrap();
        assert_eq!(u8arr.typed_data::<f32>(), vec![0.0f32, 10., 200.]);
    }

    #[test]
    fn cast_i32_u32() {
        let arr = vec![-1i32].into_array();
        let error = compute::cast::cast(&arr, PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }
}