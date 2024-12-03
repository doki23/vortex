use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{VarBinArray, VarBinEncoding};
use vortex_array::encoding::{Encoding, EncodingRef};
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct VarBinCompressor;

impl EncodingCompressor for VarBinCompressor {
    fn id(&self) -> &str {
        VarBinEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::VARBIN_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        array.is_encoding(VarBinEncoding::ID).then_some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let varbin_array = VarBinArray::try_from(array.clone())?;
        let offsets = ctx.auxiliary("offsets").compress(
            &varbin_array.offsets(),
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        let (validity, validity_path) = ctx.compress_validity(
            varbin_array.validity(),
            like.as_ref().and_then(|l| l.child(2)),
        )?;

        Ok(CompressedArray::compressed(
            VarBinArray::try_new(
                offsets.array,
                varbin_array.bytes(), // we don't compress the raw bytes
                array.dtype().clone(),
                validity,
            )?
            .into_array(),
            Some(CompressionTree::new(
                self,
                vec![offsets.path, None, validity_path],
            )),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&VarBinEncoding as EncodingRef])
    }
}
