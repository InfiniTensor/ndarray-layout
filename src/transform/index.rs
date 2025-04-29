﻿use crate::ArrayLayout;
use std::iter::zip;

/// 索引变换参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IndexArg {
    /// 索引的轴。
    pub axis: usize,
    /// 选择指定轴的第几个元素。
    pub index: usize,
}

impl<const N: usize> ArrayLayout<N> {
    /// 索引变换是选择张量指定阶上一项数据的变换，例如指定向量中的一个数、指定矩阵的一行或一列。
    /// 索引变换导致张量降阶，确定索引的阶从张量表示移除。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).index(1, 2);
    /// assert_eq!(layout.shape(), &[2, 4]);
    /// assert_eq!(layout.strides(), &[12, 1]);
    /// assert_eq!(layout.offset(), 8);
    /// ```
    pub fn index(&self, axis: usize, index: usize) -> Self {
        self.index_many(&[IndexArg { axis, index }])
    }

    /// 一次对多个阶进行索引变换。
    pub fn index_many(&self, mut args: &[IndexArg]) -> Self {
        let content = self.content();
        let mut offset = content.offset();
        let shape = content.shape();
        let iter = zip(shape, content.strides()).enumerate();

        let check = |&IndexArg { axis, index }| shape.get(axis).filter(|&&d| index < d).is_some();

        if let [first, ..] = args {
            assert!(check(first), "Invalid index arg: {first:?}");
        } else {
            return self.clone();
        }

        let mut ans = Self::with_ndim(self.ndim - args.len());
        let mut content = ans.content_mut();
        let mut j = 0;
        for (i, (&d, &s)) in iter {
            match *args {
                [IndexArg { axis, index }, ref tail @ ..] if axis == i => {
                    offset += index as isize * s;
                    if let [first, ..] = tail {
                        assert!(check(first), "Invalid index arg: {first:?}");
                        assert!(first.axis > axis, "Index args must be in ascending order");
                    }
                    args = tail;
                }
                [..] => {
                    content.set_shape(j, d);
                    content.set_stride(j, s);
                    j += 1;
                }
            }
        }
        content.set_offset(offset as _);
        ans
    }
}

#[test]
fn test() {
    let layout = ArrayLayout::<1>::new(&[2, 3, 4], &[12, 4, 1], 0);
    let layout = layout.index(1, 2);
    assert_eq!(layout.shape(), &[2, 4]);
    assert_eq!(layout.strides(), &[12, 1]);
    assert_eq!(layout.offset(), 8);

    let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    let layout = layout.index(1, 2);
    assert_eq!(layout.shape(), &[2, 4]);
    assert_eq!(layout.strides(), &[12, 1]);
    assert_eq!(layout.offset(), 12);

    let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    let layout = layout.index_many(&[]);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[12, -4, 1]);
    assert_eq!(layout.offset(), 20);

    let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    let layout = layout.index_many(&[
        IndexArg { axis: 0, index: 1 },
        IndexArg { axis: 1, index: 2 },
    ]);
    assert_eq!(layout.shape(), &[4]);
    assert_eq!(layout.strides(), &[1]);
    assert_eq!(layout.offset(), 24);
}
