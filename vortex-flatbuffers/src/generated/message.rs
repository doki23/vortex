// automatically generated by the FlatBuffers compiler, do not modify


// @generated

use crate::dtype::*;
use crate::array::*;
use crate::scalar::*;
use core::mem;
use core::cmp::Ordering;

extern crate flatbuffers;
use self::flatbuffers::{EndianScalar, Follow};

#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MIN_MESSAGE_VERSION: u8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MAX_MESSAGE_VERSION: u8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
#[allow(non_camel_case_types)]
pub const ENUM_VALUES_MESSAGE_VERSION: [MessageVersion; 1] = [
  MessageVersion::V0,
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct MessageVersion(pub u8);
#[allow(non_upper_case_globals)]
impl MessageVersion {
  pub const V0: Self = Self(0);

  pub const ENUM_MIN: u8 = 0;
  pub const ENUM_MAX: u8 = 0;
  pub const ENUM_VALUES: &'static [Self] = &[
    Self::V0,
  ];
  /// Returns the variant's name or "" if unknown.
  pub fn variant_name(self) -> Option<&'static str> {
    match self {
      Self::V0 => Some("V0"),
      _ => None,
    }
  }
}
impl core::fmt::Debug for MessageVersion {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    if let Some(name) = self.variant_name() {
      f.write_str(name)
    } else {
      f.write_fmt(format_args!("<UNKNOWN {:?}>", self.0))
    }
  }
}
impl<'a> flatbuffers::Follow<'a> for MessageVersion {
  type Inner = Self;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    let b = flatbuffers::read_scalar_at::<u8>(buf, loc);
    Self(b)
  }
}

impl flatbuffers::Push for MessageVersion {
    type Output = MessageVersion;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        flatbuffers::emplace_scalar::<u8>(dst, self.0);
    }
}

impl flatbuffers::EndianScalar for MessageVersion {
  type Scalar = u8;
  #[inline]
  fn to_little_endian(self) -> u8 {
    self.0.to_le()
  }
  #[inline]
  #[allow(clippy::wrong_self_convention)]
  fn from_little_endian(v: u8) -> Self {
    let b = u8::from_le(v);
    Self(b)
  }
}

impl<'a> flatbuffers::Verifiable for MessageVersion {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    u8::run_verifier(v, pos)
  }
}

impl flatbuffers::SimpleToVerifyInSlice for MessageVersion {}
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MIN_COMPRESSION: u8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MAX_COMPRESSION: u8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
#[allow(non_camel_case_types)]
pub const ENUM_VALUES_COMPRESSION: [Compression; 1] = [
  Compression::None,
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Compression(pub u8);
#[allow(non_upper_case_globals)]
impl Compression {
  pub const None: Self = Self(0);

  pub const ENUM_MIN: u8 = 0;
  pub const ENUM_MAX: u8 = 0;
  pub const ENUM_VALUES: &'static [Self] = &[
    Self::None,
  ];
  /// Returns the variant's name or "" if unknown.
  pub fn variant_name(self) -> Option<&'static str> {
    match self {
      Self::None => Some("None"),
      _ => None,
    }
  }
}
impl core::fmt::Debug for Compression {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    if let Some(name) = self.variant_name() {
      f.write_str(name)
    } else {
      f.write_fmt(format_args!("<UNKNOWN {:?}>", self.0))
    }
  }
}
impl<'a> flatbuffers::Follow<'a> for Compression {
  type Inner = Self;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    let b = flatbuffers::read_scalar_at::<u8>(buf, loc);
    Self(b)
  }
}

impl flatbuffers::Push for Compression {
    type Output = Compression;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        flatbuffers::emplace_scalar::<u8>(dst, self.0);
    }
}

impl flatbuffers::EndianScalar for Compression {
  type Scalar = u8;
  #[inline]
  fn to_little_endian(self) -> u8 {
    self.0.to_le()
  }
  #[inline]
  #[allow(clippy::wrong_self_convention)]
  fn from_little_endian(v: u8) -> Self {
    let b = u8::from_le(v);
    Self(b)
  }
}

impl<'a> flatbuffers::Verifiable for Compression {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    u8::run_verifier(v, pos)
  }
}

impl flatbuffers::SimpleToVerifyInSlice for Compression {}
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MIN_MESSAGE_HEADER: u8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MAX_MESSAGE_HEADER: u8 = 3;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
#[allow(non_camel_case_types)]
pub const ENUM_VALUES_MESSAGE_HEADER: [MessageHeader; 4] = [
  MessageHeader::NONE,
  MessageHeader::ArrayData,
  MessageHeader::Buffer,
  MessageHeader::DType,
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct MessageHeader(pub u8);
#[allow(non_upper_case_globals)]
impl MessageHeader {
  pub const NONE: Self = Self(0);
  pub const ArrayData: Self = Self(1);
  pub const Buffer: Self = Self(2);
  pub const DType: Self = Self(3);

  pub const ENUM_MIN: u8 = 0;
  pub const ENUM_MAX: u8 = 3;
  pub const ENUM_VALUES: &'static [Self] = &[
    Self::NONE,
    Self::ArrayData,
    Self::Buffer,
    Self::DType,
  ];
  /// Returns the variant's name or "" if unknown.
  pub fn variant_name(self) -> Option<&'static str> {
    match self {
      Self::NONE => Some("NONE"),
      Self::ArrayData => Some("ArrayData"),
      Self::Buffer => Some("Buffer"),
      Self::DType => Some("DType"),
      _ => None,
    }
  }
}
impl core::fmt::Debug for MessageHeader {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    if let Some(name) = self.variant_name() {
      f.write_str(name)
    } else {
      f.write_fmt(format_args!("<UNKNOWN {:?}>", self.0))
    }
  }
}
impl<'a> flatbuffers::Follow<'a> for MessageHeader {
  type Inner = Self;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    let b = flatbuffers::read_scalar_at::<u8>(buf, loc);
    Self(b)
  }
}

impl flatbuffers::Push for MessageHeader {
    type Output = MessageHeader;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        flatbuffers::emplace_scalar::<u8>(dst, self.0);
    }
}

impl flatbuffers::EndianScalar for MessageHeader {
  type Scalar = u8;
  #[inline]
  fn to_little_endian(self) -> u8 {
    self.0.to_le()
  }
  #[inline]
  #[allow(clippy::wrong_self_convention)]
  fn from_little_endian(v: u8) -> Self {
    let b = u8::from_le(v);
    Self(b)
  }
}

impl<'a> flatbuffers::Verifiable for MessageHeader {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    u8::run_verifier(v, pos)
  }
}

impl flatbuffers::SimpleToVerifyInSlice for MessageHeader {}
pub struct MessageHeaderUnionTableOffset {}

pub enum MessageOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct Message<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for Message<'a> {
  type Inner = Message<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> Message<'a> {
  pub const VT_VERSION: flatbuffers::VOffsetT = 4;
  pub const VT_HEADER_TYPE: flatbuffers::VOffsetT = 6;
  pub const VT_HEADER: flatbuffers::VOffsetT = 8;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    Message { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args MessageArgs
  ) -> flatbuffers::WIPOffset<Message<'bldr>> {
    let mut builder = MessageBuilder::new(_fbb);
    if let Some(x) = args.header { builder.add_header(x); }
    builder.add_header_type(args.header_type);
    builder.add_version(args.version);
    builder.finish()
  }


  #[inline]
  pub fn version(&self) -> MessageVersion {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<MessageVersion>(Message::VT_VERSION, Some(MessageVersion::V0)).unwrap()}
  }
  #[inline]
  pub fn header_type(&self) -> MessageHeader {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<MessageHeader>(Message::VT_HEADER_TYPE, Some(MessageHeader::NONE)).unwrap()}
  }
  #[inline]
  pub fn header(&self) -> Option<flatbuffers::Table<'a>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Table<'a>>>(Message::VT_HEADER, None)}
  }
  #[inline]
  #[allow(non_snake_case)]
  pub fn header_as_array_data(&self) -> Option<ArrayData<'a>> {
    if self.header_type() == MessageHeader::ArrayData {
      self.header().map(|t| {
       // Safety:
       // Created from a valid Table for this object
       // Which contains a valid union in this slot
       unsafe { ArrayData::init_from_table(t) }
     })
    } else {
      None
    }
  }

  #[inline]
  #[allow(non_snake_case)]
  pub fn header_as_buffer(&self) -> Option<Buffer<'a>> {
    if self.header_type() == MessageHeader::Buffer {
      self.header().map(|t| {
       // Safety:
       // Created from a valid Table for this object
       // Which contains a valid union in this slot
       unsafe { Buffer::init_from_table(t) }
     })
    } else {
      None
    }
  }

  #[inline]
  #[allow(non_snake_case)]
  pub fn header_as_dtype(&self) -> Option<DType<'a>> {
    if self.header_type() == MessageHeader::DType {
      self.header().map(|t| {
       // Safety:
       // Created from a valid Table for this object
       // Which contains a valid union in this slot
       unsafe { DType::init_from_table(t) }
     })
    } else {
      None
    }
  }

}

impl flatbuffers::Verifiable for Message<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<MessageVersion>("version", Self::VT_VERSION, false)?
     .visit_union::<MessageHeader, _>("header_type", Self::VT_HEADER_TYPE, "header", Self::VT_HEADER, false, |key, v, pos| {
        match key {
          MessageHeader::ArrayData => v.verify_union_variant::<flatbuffers::ForwardsUOffset<ArrayData>>("MessageHeader::ArrayData", pos),
          MessageHeader::Buffer => v.verify_union_variant::<flatbuffers::ForwardsUOffset<Buffer>>("MessageHeader::Buffer", pos),
          MessageHeader::DType => v.verify_union_variant::<flatbuffers::ForwardsUOffset<DType>>("MessageHeader::DType", pos),
          _ => Ok(()),
        }
     })?
     .finish();
    Ok(())
  }
}
pub struct MessageArgs {
    pub version: MessageVersion,
    pub header_type: MessageHeader,
    pub header: Option<flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>>,
}
impl<'a> Default for MessageArgs {
  #[inline]
  fn default() -> Self {
    MessageArgs {
      version: MessageVersion::V0,
      header_type: MessageHeader::NONE,
      header: None,
    }
  }
}

pub struct MessageBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> MessageBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_version(&mut self, version: MessageVersion) {
    self.fbb_.push_slot::<MessageVersion>(Message::VT_VERSION, version, MessageVersion::V0);
  }
  #[inline]
  pub fn add_header_type(&mut self, header_type: MessageHeader) {
    self.fbb_.push_slot::<MessageHeader>(Message::VT_HEADER_TYPE, header_type, MessageHeader::NONE);
  }
  #[inline]
  pub fn add_header(&mut self, header: flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(Message::VT_HEADER, header);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> MessageBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    MessageBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<Message<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for Message<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Message");
      ds.field("version", &self.version());
      ds.field("header_type", &self.header_type());
      match self.header_type() {
        MessageHeader::ArrayData => {
          if let Some(x) = self.header_as_array_data() {
            ds.field("header", &x)
          } else {
            ds.field("header", &"InvalidFlatbuffer: Union discriminant does not match value.")
          }
        },
        MessageHeader::Buffer => {
          if let Some(x) = self.header_as_buffer() {
            ds.field("header", &x)
          } else {
            ds.field("header", &"InvalidFlatbuffer: Union discriminant does not match value.")
          }
        },
        MessageHeader::DType => {
          if let Some(x) = self.header_as_dtype() {
            ds.field("header", &x)
          } else {
            ds.field("header", &"InvalidFlatbuffer: Union discriminant does not match value.")
          }
        },
        _ => {
          let x: Option<()> = None;
          ds.field("header", &x)
        },
      };
      ds.finish()
  }
}
#[inline]
/// Verifies that a buffer of bytes contains a `Message`
/// and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_message_unchecked`.
pub fn root_as_message(buf: &[u8]) -> Result<Message, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root::<Message>(buf)
}
#[inline]
/// Verifies that a buffer of bytes contains a size prefixed
/// `Message` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `size_prefixed_root_as_message_unchecked`.
pub fn size_prefixed_root_as_message(buf: &[u8]) -> Result<Message, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root::<Message>(buf)
}
#[inline]
/// Verifies, with the given options, that a buffer of bytes
/// contains a `Message` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_message_unchecked`.
pub fn root_as_message_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<Message<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root_with_opts::<Message<'b>>(opts, buf)
}
#[inline]
/// Verifies, with the given verifier options, that a buffer of
/// bytes contains a size prefixed `Message` and returns
/// it. Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_message_unchecked`.
pub fn size_prefixed_root_as_message_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<Message<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root_with_opts::<Message<'b>>(opts, buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a Message and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid `Message`.
pub unsafe fn root_as_message_unchecked(buf: &[u8]) -> Message {
  flatbuffers::root_unchecked::<Message>(buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a size prefixed Message and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid size prefixed `Message`.
pub unsafe fn size_prefixed_root_as_message_unchecked(buf: &[u8]) -> Message {
  flatbuffers::size_prefixed_root_unchecked::<Message>(buf)
}
#[inline]
pub fn finish_message_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    root: flatbuffers::WIPOffset<Message<'a>>) {
  fbb.finish(root, None);
}

#[inline]
pub fn finish_size_prefixed_message_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>, root: flatbuffers::WIPOffset<Message<'a>>) {
  fbb.finish_size_prefixed(root, None);
}
