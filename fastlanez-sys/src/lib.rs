#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::mem::{size_of, MaybeUninit};

use arrayref::array_mut_ref;
use seq_macro::seq;
use uninit::prelude::VecCapacity;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct Pred<const B: bool>;

pub trait Satisfied {}

impl Satisfied for Pred<true> {}

/// BitPack into a compile-time known bit-width.
pub trait BitPack<const W: usize>
where
    Self: Sized,
    Pred<{ W > 0 }>: Satisfied,
    Pred<{ W < 8 * size_of::<Self>() }>: Satisfied,
{
    fn bitpack<'a>(
        input: &[Self; 1024],
        output: &'a mut [MaybeUninit<u8>; 128 * W],
    ) -> &'a [u8; 128 * W];
}

#[derive(Debug)]
pub struct UnsupportedBitWidth;

/// Try to bitpack into a runtime-known bit width.
pub trait TryBitPack
where
    Self: Sized,
{
    fn try_bitpack<'a>(
        input: &[Self; 1024],
        width: usize,
        output: &'a mut [MaybeUninit<u8>],
    ) -> Result<&'a [u8], UnsupportedBitWidth>;

    fn try_bitpack_into(
        input: &[Self; 1024],
        width: usize,
        output: &mut Vec<u8>,
    ) -> Result<(), UnsupportedBitWidth> {
        Self::try_bitpack(input, width, output.reserve_uninit(width * 128))?;
        unsafe { output.set_len(output.len() + (width * 128)) }
        Ok(())
    }
}

macro_rules! bitpack_impl {
    ($T:ty, $W:literal) => {
        paste::item! {
            seq!(N in 1..$W {
                impl BitPack<N> for $T {
                    #[inline]
                    fn bitpack<'a>(
                        input: &[Self; 1024],
                        output: &'a mut [MaybeUninit<u8>; 128 * N],
                    ) -> &'a [u8; 128 * N] {
                            unsafe {
                                let output_array: &mut [u8; 128 * N] = std::mem::transmute(output);
                                [<fl_bitpack_ $T _u >]~N(input, output_array);
                                output_array
                            }
                    }
                }
            });
        }

        impl TryBitPack for $T {
            fn try_bitpack<'a>(
                input: &[Self; 1024],
                width: usize,
                output: &'a mut [MaybeUninit<u8>],
            ) -> Result<&'a [u8], UnsupportedBitWidth> {
                seq!(N in 1..$W {
                    match width {
                        #(N => Ok(BitPack::<N>::bitpack(input, array_mut_ref![output, 0, N * 128]).as_slice()),)*
                        _ => Err(UnsupportedBitWidth),
                    }
                })
            }
        }
    };
}

bitpack_impl!(u8, 8);
bitpack_impl!(u16, 16);
bitpack_impl!(u32, 32);
bitpack_impl!(u64, 64);