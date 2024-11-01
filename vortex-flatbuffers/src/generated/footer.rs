// automatically generated by the FlatBuffers compiler, do not modify


// @generated

use core::mem;
use core::cmp::Ordering;

extern crate flatbuffers;
use self::flatbuffers::{EndianScalar, Follow};

// struct Buffer, aligned to 8
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq)]
pub struct Buffer(pub [u8; 16]);
impl Default for Buffer { 
  fn default() -> Self { 
    Self([0; 16])
  }
}
impl core::fmt::Debug for Buffer {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    f.debug_struct("Buffer")
      .field("begin", &self.begin())
      .field("end", &self.end())
      .finish()
  }
}

impl flatbuffers::SimpleToVerifyInSlice for Buffer {}
impl<'a> flatbuffers::Follow<'a> for Buffer {
  type Inner = &'a Buffer;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    <&'a Buffer>::follow(buf, loc)
  }
}
impl<'a> flatbuffers::Follow<'a> for &'a Buffer {
  type Inner = &'a Buffer;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    flatbuffers::follow_cast_ref::<Buffer>(buf, loc)
  }
}
impl<'b> flatbuffers::Push for Buffer {
    type Output = Buffer;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        let src = ::core::slice::from_raw_parts(self as *const Buffer as *const u8, Self::size());
        dst.copy_from_slice(src);
    }
}

impl<'a> flatbuffers::Verifiable for Buffer {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.in_buffer::<Self>(pos)
  }
}

impl<'a> Buffer {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    begin: u64,
    end: u64,
  ) -> Self {
    let mut s = Self([0; 16]);
    s.set_begin(begin);
    s.set_end(end);
    s
  }

  pub fn begin(&self) -> u64 {
    let mut mem = core::mem::MaybeUninit::<<u64 as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[0..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<u64 as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_begin(&mut self, x: u64) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[0..].as_mut_ptr(),
        core::mem::size_of::<<u64 as EndianScalar>::Scalar>(),
      );
    }
  }

  pub fn end(&self) -> u64 {
    let mut mem = core::mem::MaybeUninit::<<u64 as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[8..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<u64 as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_end(&mut self, x: u64) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[8..].as_mut_ptr(),
        core::mem::size_of::<<u64 as EndianScalar>::Scalar>(),
      );
    }
  }

}

pub enum LayoutOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct Layout<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for Layout<'a> {
  type Inner = Layout<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> Layout<'a> {
  pub const VT_ENCODING: flatbuffers::VOffsetT = 4;
  pub const VT_BUFFERS: flatbuffers::VOffsetT = 6;
  pub const VT_CHILDREN: flatbuffers::VOffsetT = 8;
  pub const VT_LENGTH: flatbuffers::VOffsetT = 10;
  pub const VT_METADATA: flatbuffers::VOffsetT = 12;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    Layout { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args LayoutArgs<'args>
  ) -> flatbuffers::WIPOffset<Layout<'bldr>> {
    let mut builder = LayoutBuilder::new(_fbb);
    builder.add_length(args.length);
    if let Some(x) = args.metadata { builder.add_metadata(x); }
    if let Some(x) = args.children { builder.add_children(x); }
    if let Some(x) = args.buffers { builder.add_buffers(x); }
    builder.add_encoding(args.encoding);
    builder.finish()
  }


  #[inline]
  pub fn encoding(&self) -> u16 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<u16>(Layout::VT_ENCODING, Some(0)).unwrap()}
  }
  #[inline]
  pub fn buffers(&self) -> Option<flatbuffers::Vector<'a, Buffer>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, Buffer>>>(Layout::VT_BUFFERS, None)}
  }
  #[inline]
  pub fn children(&self) -> Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Layout<'a>>>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Layout>>>>(Layout::VT_CHILDREN, None)}
  }
  #[inline]
  pub fn length(&self) -> u64 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<u64>(Layout::VT_LENGTH, Some(0)).unwrap()}
  }
  #[inline]
  pub fn metadata(&self) -> Option<flatbuffers::Vector<'a, u8>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, u8>>>(Layout::VT_METADATA, None)}
  }
}

impl flatbuffers::Verifiable for Layout<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<u16>("encoding", Self::VT_ENCODING, false)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, Buffer>>>("buffers", Self::VT_BUFFERS, false)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Layout>>>>("children", Self::VT_CHILDREN, false)?
     .visit_field::<u64>("length", Self::VT_LENGTH, false)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, u8>>>("metadata", Self::VT_METADATA, false)?
     .finish();
    Ok(())
  }
}
pub struct LayoutArgs<'a> {
    pub encoding: u16,
    pub buffers: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, Buffer>>>,
    pub children: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Layout<'a>>>>>,
    pub length: u64,
    pub metadata: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, u8>>>,
}
impl<'a> Default for LayoutArgs<'a> {
  #[inline]
  fn default() -> Self {
    LayoutArgs {
      encoding: 0,
      buffers: None,
      children: None,
      length: 0,
      metadata: None,
    }
  }
}

pub struct LayoutBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> LayoutBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_encoding(&mut self, encoding: u16) {
    self.fbb_.push_slot::<u16>(Layout::VT_ENCODING, encoding, 0);
  }
  #[inline]
  pub fn add_buffers(&mut self, buffers: flatbuffers::WIPOffset<flatbuffers::Vector<'b , Buffer>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(Layout::VT_BUFFERS, buffers);
  }
  #[inline]
  pub fn add_children(&mut self, children: flatbuffers::WIPOffset<flatbuffers::Vector<'b , flatbuffers::ForwardsUOffset<Layout<'b >>>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(Layout::VT_CHILDREN, children);
  }
  #[inline]
  pub fn add_length(&mut self, length: u64) {
    self.fbb_.push_slot::<u64>(Layout::VT_LENGTH, length, 0);
  }
  #[inline]
  pub fn add_metadata(&mut self, metadata: flatbuffers::WIPOffset<flatbuffers::Vector<'b , u8>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(Layout::VT_METADATA, metadata);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> LayoutBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    LayoutBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<Layout<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for Layout<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Layout");
      ds.field("encoding", &self.encoding());
      ds.field("buffers", &self.buffers());
      ds.field("children", &self.children());
      ds.field("length", &self.length());
      ds.field("metadata", &self.metadata());
      ds.finish()
  }
}
pub enum FooterOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct Footer<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for Footer<'a> {
  type Inner = Footer<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> Footer<'a> {
  pub const VT_LAYOUT: flatbuffers::VOffsetT = 4;
  pub const VT_ROW_COUNT: flatbuffers::VOffsetT = 6;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    Footer { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args FooterArgs<'args>
  ) -> flatbuffers::WIPOffset<Footer<'bldr>> {
    let mut builder = FooterBuilder::new(_fbb);
    builder.add_row_count(args.row_count);
    if let Some(x) = args.layout { builder.add_layout(x); }
    builder.finish()
  }


  #[inline]
  pub fn layout(&self) -> Option<Layout<'a>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<Layout>>(Footer::VT_LAYOUT, None)}
  }
  #[inline]
  pub fn row_count(&self) -> u64 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<u64>(Footer::VT_ROW_COUNT, Some(0)).unwrap()}
  }
}

impl flatbuffers::Verifiable for Footer<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<flatbuffers::ForwardsUOffset<Layout>>("layout", Self::VT_LAYOUT, false)?
     .visit_field::<u64>("row_count", Self::VT_ROW_COUNT, false)?
     .finish();
    Ok(())
  }
}
pub struct FooterArgs<'a> {
    pub layout: Option<flatbuffers::WIPOffset<Layout<'a>>>,
    pub row_count: u64,
}
impl<'a> Default for FooterArgs<'a> {
  #[inline]
  fn default() -> Self {
    FooterArgs {
      layout: None,
      row_count: 0,
    }
  }
}

pub struct FooterBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> FooterBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_layout(&mut self, layout: flatbuffers::WIPOffset<Layout<'b >>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<Layout>>(Footer::VT_LAYOUT, layout);
  }
  #[inline]
  pub fn add_row_count(&mut self, row_count: u64) {
    self.fbb_.push_slot::<u64>(Footer::VT_ROW_COUNT, row_count, 0);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> FooterBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    FooterBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<Footer<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for Footer<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Footer");
      ds.field("layout", &self.layout());
      ds.field("row_count", &self.row_count());
      ds.finish()
  }
}
pub enum PostscriptOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct Postscript<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for Postscript<'a> {
  type Inner = Postscript<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> Postscript<'a> {
  pub const VT_SCHEMA_OFFSET: flatbuffers::VOffsetT = 4;
  pub const VT_FOOTER_OFFSET: flatbuffers::VOffsetT = 6;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    Postscript { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args PostscriptArgs
  ) -> flatbuffers::WIPOffset<Postscript<'bldr>> {
    let mut builder = PostscriptBuilder::new(_fbb);
    builder.add_footer_offset(args.footer_offset);
    builder.add_schema_offset(args.schema_offset);
    builder.finish()
  }


  #[inline]
  pub fn schema_offset(&self) -> u64 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<u64>(Postscript::VT_SCHEMA_OFFSET, Some(0)).unwrap()}
  }
  #[inline]
  pub fn footer_offset(&self) -> u64 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<u64>(Postscript::VT_FOOTER_OFFSET, Some(0)).unwrap()}
  }
}

impl flatbuffers::Verifiable for Postscript<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<u64>("schema_offset", Self::VT_SCHEMA_OFFSET, false)?
     .visit_field::<u64>("footer_offset", Self::VT_FOOTER_OFFSET, false)?
     .finish();
    Ok(())
  }
}
pub struct PostscriptArgs {
    pub schema_offset: u64,
    pub footer_offset: u64,
}
impl<'a> Default for PostscriptArgs {
  #[inline]
  fn default() -> Self {
    PostscriptArgs {
      schema_offset: 0,
      footer_offset: 0,
    }
  }
}

pub struct PostscriptBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> PostscriptBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_schema_offset(&mut self, schema_offset: u64) {
    self.fbb_.push_slot::<u64>(Postscript::VT_SCHEMA_OFFSET, schema_offset, 0);
  }
  #[inline]
  pub fn add_footer_offset(&mut self, footer_offset: u64) {
    self.fbb_.push_slot::<u64>(Postscript::VT_FOOTER_OFFSET, footer_offset, 0);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> PostscriptBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    PostscriptBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<Postscript<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for Postscript<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Postscript");
      ds.field("schema_offset", &self.schema_offset());
      ds.field("footer_offset", &self.footer_offset());
      ds.finish()
  }
}
#[inline]
/// Verifies that a buffer of bytes contains a `Footer`
/// and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_footer_unchecked`.
pub fn root_as_footer(buf: &[u8]) -> Result<Footer, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root::<Footer>(buf)
}
#[inline]
/// Verifies that a buffer of bytes contains a size prefixed
/// `Footer` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `size_prefixed_root_as_footer_unchecked`.
pub fn size_prefixed_root_as_footer(buf: &[u8]) -> Result<Footer, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root::<Footer>(buf)
}
#[inline]
/// Verifies, with the given options, that a buffer of bytes
/// contains a `Footer` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_footer_unchecked`.
pub fn root_as_footer_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<Footer<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root_with_opts::<Footer<'b>>(opts, buf)
}
#[inline]
/// Verifies, with the given verifier options, that a buffer of
/// bytes contains a size prefixed `Footer` and returns
/// it. Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_footer_unchecked`.
pub fn size_prefixed_root_as_footer_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<Footer<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root_with_opts::<Footer<'b>>(opts, buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a Footer and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid `Footer`.
pub unsafe fn root_as_footer_unchecked(buf: &[u8]) -> Footer {
  flatbuffers::root_unchecked::<Footer>(buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a size prefixed Footer and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid size prefixed `Footer`.
pub unsafe fn size_prefixed_root_as_footer_unchecked(buf: &[u8]) -> Footer {
  flatbuffers::size_prefixed_root_unchecked::<Footer>(buf)
}
#[inline]
pub fn finish_footer_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    root: flatbuffers::WIPOffset<Footer<'a>>) {
  fbb.finish(root, None);
}

#[inline]
pub fn finish_size_prefixed_footer_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>, root: flatbuffers::WIPOffset<Footer<'a>>) {
  fbb.finish_size_prefixed(root, None);
}
