use std::io;

use crate::array::varbinview::{VarBinViewArray, VarBinViewEncoding};
use crate::array::{Array, ArrayRef};
use crate::serde::{ArraySerde, EncodingSerde, ReadCtx, WriteCtx};

impl ArraySerde for VarBinViewArray {
    fn write(&self, ctx: &mut WriteCtx) -> io::Result<()> {
        if let Some(v) = self.validity() {
            ctx.write(v.as_ref())?;
        }
        ctx.write(self.views())?;
        ctx.write_usize(self.data().len())?;
        for d in self.data() {
            ctx.write(d.as_ref())?;
        }
        Ok(())
    }
}

impl EncodingSerde for VarBinViewEncoding {
    fn read(&self, ctx: &mut ReadCtx) -> io::Result<ArrayRef> {
        let validity = if ctx.schema().is_nullable() {
            Some(ctx.validity().read()?)
        } else {
            None
        };
        let views = ctx.bytes().read()?;
        let num_data = ctx.read_usize()?;
        let mut data_bufs = Vec::<ArrayRef>::with_capacity(num_data);
        for _ in 0..num_data {
            data_bufs.push(ctx.bytes().read()?);
        }
        Ok(VarBinViewArray::new(views, data_bufs, ctx.schema().clone(), validity).boxed())
    }
}

#[cfg(test)]
mod test {
    use crate::array::downcast::DowncastArrayBuiltin;
    use crate::array::primitive::PrimitiveArray;
    use crate::array::varbinview::{BinaryView, Inlined, Ref, VarBinViewArray};
    use crate::array::Array;
    use crate::dtype::{DType, Nullability};
    use crate::serde::test::roundtrip_array;

    fn binary_array() -> VarBinViewArray {
        let values = PrimitiveArray::from("hello world this is a long string".as_bytes().to_vec());
        let view1 = BinaryView {
            inlined: Inlined::new("hello world"),
        };
        let view2 = BinaryView {
            _ref: Ref {
                size: 33,
                prefix: "hell".as_bytes().try_into().unwrap(),
                buffer_index: 0,
                offset: 0,
            },
        };
        let view_arr = PrimitiveArray::from(
            vec![view1.to_le_bytes(), view2.to_le_bytes()]
                .into_iter()
                .flatten()
                .collect::<Vec<u8>>(),
        );

        VarBinViewArray::new(
            view_arr.boxed(),
            vec![values.boxed()],
            DType::Utf8(Nullability::NonNullable),
            None,
        )
    }

    #[test]
    fn roundtrip() {
        let arr = binary_array();
        let read_arr = roundtrip_array(arr.as_ref()).unwrap();

        assert_eq!(
            arr.views().as_primitive().buffer().typed_data::<u8>(),
            read_arr
                .as_varbinview()
                .views()
                .as_primitive()
                .buffer()
                .typed_data::<u8>()
        );

        assert_eq!(
            arr.data()[0].as_primitive().buffer().typed_data::<u8>(),
            read_arr.as_varbinview().data()[0]
                .as_primitive()
                .buffer()
                .typed_data::<u8>()
        );
    }
}