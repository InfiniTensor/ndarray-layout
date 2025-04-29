﻿use crate::ArrayLayout;

/// 切分变换参数。
pub struct Split<'a, const N: usize> {
    src: &'a ArrayLayout<N>,
    axis: usize,
    start: usize,
    parts: &'a [usize],
}

impl<const N: usize> ArrayLayout<N> {
    /// 切分变换讲单个张量沿某个维度切分成多个张量，因此可以支持不均匀的切分。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    /// let mut splits = layout.split(2, &[1, 3]);
    ///
    /// let layout = splits.next().unwrap();
    /// assert_eq!(layout.shape(), &[2, 3, 1]);
    /// assert_eq!(layout.strides(), &[12, 4, 1]);
    /// assert_eq!(layout.offset(), 0);
    ///
    /// let layout = splits.next().unwrap();
    /// assert_eq!(layout.shape(), &[2, 3, 3]);
    /// assert_eq!(layout.strides(), &[12, 4, 1]);
    /// assert_eq!(layout.offset(), 1);
    /// ```
    #[inline]
    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> Split<'a, N> {
        assert_eq!(self.shape()[axis], parts.iter().sum());
        Split {
            src: self,
            axis,
            start: 0,
            parts,
        }
    }
}

impl<const N: usize> Iterator for Split<'_, N> {
    type Item = ArrayLayout<N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parts.split_first().map(|(&head, tail)| {
            let start = self.start;
            self.start += head;
            self.parts = tail;
            self.src.slice(self.axis, start, 1, head)
        })
    }
}

#[test]
fn test_split() {
    let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    let mut splits = layout.split(2, &[1, 3]);
    let layout = splits.next().unwrap();
    assert_eq!(layout.shape(), &[2, 3, 1]);
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert_eq!(layout.offset(), 0);
    let layout = splits.next().unwrap();
    assert_eq!(layout.shape(), &[2, 3, 3]);
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert_eq!(layout.offset(), 1);
}
