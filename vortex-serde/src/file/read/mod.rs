use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::Array;
use vortex_error::VortexResult;

mod buffered;
pub mod builder;
mod cache;
mod column_batch;
mod context;
mod expr_project;
mod filtering;
pub mod layouts;
mod mask;
mod recordbatchreader;
mod stream;

pub use builder::initial_read::*;
pub use builder::VortexReadBuilder;
pub use cache::*;
pub use context::*;
pub use filtering::RowFilter;
pub use recordbatchreader::{AsyncRuntime, VortexRecordBatchReader};
pub use stream::VortexFileArrayStream;
use vortex_expr::VortexExpr;
pub use vortex_schema::projection::Projection;
pub use vortex_schema::Schema;

pub use crate::file::read::mask::RowMask;
use crate::stream_writer::ByteRange;

// Recommended read-size according to the AWS performance guide
pub const INITIAL_READ_SIZE: usize = 8 * 1024 * 1024;

/// Operation to apply to data returned by the layout
#[derive(Debug, Clone)]
pub struct Scan {
    expr: Option<Arc<dyn VortexExpr>>,
}

impl Scan {
    pub fn new(expr: Option<Arc<dyn VortexExpr>>) -> Self {
        Self { expr }
    }
}

/// Unique identifier for a message within a layout
pub type LayoutPartId = u16;
/// Path through layout tree to given message
pub type MessageId = Vec<LayoutPartId>;
/// ID and Range of atomic element of the file
pub type Message = (MessageId, ByteRange);

#[derive(Debug)]
pub enum BatchRead {
    ReadMore(Vec<Message>),
    Batch(Array),
}

/// A reader for a layout, a serialized sequence of Vortex arrays.
///
/// Some layouts are _horizontally divisble_: they can read a sub-sequence of rows independently of
/// other sub-sequences. A layout advertises its sub-divisions in its [add_splits][Self::add_splits]
/// method. Any layout which is or contains a chunked layout is horizontally divisble.
///
/// The [read_selection][Self::read_selection] method accepts and applies a [RowMask], reading only
/// the sub-divisions which contain the selected (i.e. masked) rows.
pub trait LayoutReader: Debug + Send {
    /// Register all horizontal row boundaries of this layout.
    ///
    /// Layout should register all indivisible absolute row boundaries of the data stored in itself and its children.
    /// `row_offset` gives the relative row position of this layout to the beginning of the file.
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()>;

    /// Reads the data from the underlying layout within given selection
    ///
    /// Layout is required to return all data for given selection in one batch.  Layout can either
    /// return a batch of data (i.e., an Array) or ask for more layout messages to be read. When
    /// requesting messages to be read the caller should populate the message cache used when
    /// creating the invoked instance of this trait and then call back into this function.
    ///
    /// The layout is finished producing data for selection when it returns None
    fn read_selection(&mut self, selector: &RowMask) -> VortexResult<Option<BatchRead>>;
}