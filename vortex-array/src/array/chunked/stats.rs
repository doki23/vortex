use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::stats::{ArrayStatistics, ArrayStatisticsCompute, Stat, StatsSet};

impl ArrayStatisticsCompute for ChunkedArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        // for UncompressedSizeInBytes, we end up with sum of chunk uncompressed sizes
        // this ignores the `chunk_offsets` array child, so it won't exactly match self.nbytes()
        Ok(self
            .chunks()
            .map(|c| {
                let s = c.statistics();
                s.compute(stat);
                s.to_set()
            })
            .reduce(|mut acc, x| {
                acc.merge_ordered(&x);
                acc
            })
            .unwrap_or_default())
    }
}
