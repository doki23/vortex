// (c) Copyright 2024 Fulcrum Technologies, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::array::constant::ConstantArray;
use crate::array::{Array, ArrayKind, ArrayRef};
use crate::error::{VortexError, VortexResult};
use crate::scalar::Scalar;

// TODO(ngates): convert this to arithmetic operations with macro over the kernel.
pub fn add(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    // Check that the arrays are the same length.
    let length = lhs.len();
    if rhs.len() != length {
        return Err(VortexError::LengthMismatch);
    }

    match (ArrayKind::from(lhs), ArrayKind::from(rhs)) {
        (ArrayKind::Constant(lhs), ArrayKind::Constant(rhs)) => {
            Ok(ConstantArray::new(add_scalars(lhs.scalar(), rhs.scalar())?, length).boxed())
        }
        (ArrayKind::Constant(lhs), _) => add_scalar(rhs, lhs.scalar()),
        (_, ArrayKind::Constant(rhs)) => add_scalar(lhs, rhs.scalar()),
        _ => todo!("Implement default addition"),
    }
}

pub fn add_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> VortexResult<ArrayRef> {
    match ArrayKind::from(lhs) {
        ArrayKind::Constant(lhs) => {
            Ok(ConstantArray::new(add_scalars(lhs.scalar(), rhs)?, lhs.len()).boxed())
        }
        _ => todo!("Implement default addition"),
    }
}

pub fn add_scalars(_lhs: &dyn Scalar, _rhs: &dyn Scalar) -> VortexResult<Box<dyn Scalar>> {
    // Might need to improve this implementation...
    Ok(24.into())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let lhs = ConstantArray::new(47.into(), 100);
        let rhs = ConstantArray::new(47.into(), 100);
        let result = add(&lhs, &rhs).unwrap();
        assert_eq!(result.len(), 100);
        // assert_eq!(result.scalar_at(0), 94);
    }
}