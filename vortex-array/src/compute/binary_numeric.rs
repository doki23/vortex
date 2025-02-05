use std::sync::Arc;

use arrow_array::ArrayRef;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::{BinaryNumericOperator, Scalar};

use crate::array::ConstantArray;
use crate::arrow::{Datum, FromArrowArray};
use crate::encoding::{downcast_array_ref, Encoding};
use crate::{ArrayDType, ArrayData, IntoArrayData as _};

pub trait BinaryNumericFn<Array> {
    fn binary_numeric(
        &self,
        array: &Array,
        other: &ArrayData,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayData>>;
}

impl<E: Encoding> BinaryNumericFn<ArrayData> for E
where
    E: BinaryNumericFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn binary_numeric(
        &self,
        lhs: &ArrayData,
        rhs: &ArrayData,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let (array_ref, encoding) = downcast_array_ref::<E>(lhs)?;
        BinaryNumericFn::binary_numeric(encoding, array_ref, rhs, op)
    }
}

/// Point-wise add two numeric arrays.
pub fn add(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
pub fn add_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
pub fn sub(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
pub fn sub_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
pub fn mul(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
pub fn mul_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
pub fn div(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
pub fn div_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

pub fn binary_numeric(
    lhs: &ArrayData,
    rhs: &ArrayData,
    op: BinaryNumericOperator,
) -> VortexResult<ArrayData> {
    if lhs.len() != rhs.len() {
        vortex_bail!("Numeric operations aren't supported on arrays of different lengths")
    }
    if !matches!(lhs.dtype(), DType::Primitive(_, _))
        || !matches!(rhs.dtype(), DType::Primitive(_, _))
        || lhs.dtype() != rhs.dtype()
    {
        vortex_bail!(
            "Numeric operations are only supported on two arrays sharing the same primitive-type: {} {}",
            lhs.dtype(),
            rhs.dtype()
        )
    }

    // Check if LHS supports the operation directly.
    if let Some(fun) = lhs.encoding().binary_numeric_fn() {
        if let Some(result) = fun.binary_numeric(lhs, rhs, op)? {
            debug_assert_eq!(
                result.len(),
                lhs.len(),
                "Numeric operation length mismatch {}",
                lhs.encoding().id()
            );
            debug_assert_eq!(
                result.dtype(),
                &DType::Primitive(
                    PType::try_from(lhs.dtype())?,
                    (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()
                ),
                "Numeric operation dtype mismatch {}",
                lhs.encoding().id()
            );
            return Ok(result);
        }
    }

    // Check if RHS supports the operation directly.
    if let Some(fun) = rhs.encoding().binary_numeric_fn() {
        if let Some(result) = fun.binary_numeric(rhs, lhs, op)? {
            debug_assert_eq!(
                result.len(),
                lhs.len(),
                "Numeric operation length mismatch {}",
                rhs.encoding().id()
            );
            debug_assert_eq!(
                result.dtype(),
                &DType::Primitive(
                    PType::try_from(lhs.dtype())?,
                    (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()
                ),
                "Numeric operation dtype mismatch {}",
                rhs.encoding().id()
            );
            return Ok(result);
        }
    }

    log::debug!(
        "No numeric implementation found for LHS {}, RHS {}, and operator {:?}",
        lhs.encoding().id(),
        rhs.encoding().id(),
        op,
    );

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_numeric(lhs.clone(), rhs.clone(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
fn arrow_numeric(
    lhs: ArrayData,
    rhs: ArrayData,
    operator: BinaryNumericOperator,
) -> VortexResult<ArrayData> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = Datum::try_from(lhs)?;
    let rhs = Datum::try_from(rhs)?;

    let array = match operator {
        BinaryNumericOperator::Add => arrow_arith::numeric::add(&lhs, &rhs)?,
        BinaryNumericOperator::Sub => arrow_arith::numeric::sub(&lhs, &rhs)?,
        BinaryNumericOperator::Div => arrow_arith::numeric::div(&lhs, &rhs)?,
        BinaryNumericOperator::Mul => arrow_arith::numeric::mul(&lhs, &rhs)?,
    };

    Ok(ArrayData::from_arrow(Arc::new(array) as ArrayRef, nullable))
}

#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::array::PrimitiveArray;
    use crate::compute::{scalar_at, sub_scalar};
    use crate::{ArrayLen as _, IntoArrayData, IntoCanonical};

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = vec![1u16, 2, 3].into_array();
        let results = sub_scalar(&values, 1u16.into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<u16>()
            .to_vec();
        assert_eq!(results, &[0u16, 1, 2]);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = vec![1i64, 2, 3].into_array();
        let results = sub_scalar(&values, (-1i64).into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<i64>()
            .to_vec();
        assert_eq!(results, &[2i64, 3, 4]);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_nullable_vec(vec![Some(1u16), Some(2), None, Some(3)])
            .into_array();
        let result = sub_scalar(&values, Some(1u16).into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();

        let actual = (0..result.len())
            .map(|index| scalar_at(&result, index).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                Scalar::from(Some(0u16)),
                Scalar::from(Some(1u16)),
                Scalar::from(None::<u16>),
                Scalar::from(Some(2u16))
            ]
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = vec![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let results = sub_scalar(&values, to_subtract.into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<f64>()
            .to_vec();
        assert_eq!(results, &[2.0f64, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = vec![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = sub_scalar(&values, f32::MAX.into()).unwrap();
    }

    #[test]
    fn test_scalar_subtract_type_mismatch_fails() {
        let values = vec![1u64, 2, 3].into_array();
        // Subtracting incompatible dtypes should fail
        let _results =
            sub_scalar(&values, 1.5f64.into()).expect_err("Expected type mismatch error");
    }
}
