﻿use crate::ArrayLayout;
use std::{collections::BTreeSet, iter::zip};

impl<const N: usize> ArrayLayout<N> {
    /// 转置变换允许调换张量的维度顺序，但不改变元素的存储顺序。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[1, 0]);
    /// assert_eq!(layout.shape(), &[3, 2, 4]);
    /// assert_eq!(layout.strides(), &[4, 12, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> Self {
        let perm_ = perm.iter().collect::<BTreeSet<_>>();
        assert_eq!(perm_.len(), perm.len());

        let content = self.content();
        let shape = content.shape();
        let strides = content.strides();

        let mut ans = Self::with_ndim(self.ndim);
        let mut content = ans.content_mut();
        content.set_offset(self.offset());
        let mut set = |i, j| {
            content.set_shape(i, shape[j]);
            content.set_stride(i, strides[j]);
        };

        let mut last = 0;
        for (&i, &j) in zip(perm_, perm) {
            for i in last..i {
                set(i, i);
            }
            set(i, j);
            last = i + 1;
        }
        for i in last..shape.len() {
            set(i, i);
        }
        ans
    }
}

#[test]
fn test_transpose() {
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[1, 0]);
    assert_eq!(layout.shape(), &[3, 2, 4]);
    assert_eq!(layout.strides(), &[4, 12, 1]);
    assert_eq!(layout.offset(), 0);

    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).transpose(&[2, 0]);
    assert_eq!(layout.shape(), &[4, 3, 2]);
    assert_eq!(layout.strides(), &[1, 4, 12]);
    assert_eq!(layout.offset(), 0);
}
