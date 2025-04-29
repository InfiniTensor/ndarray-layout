use crate::{ArrayLayout, Endian};
use std::iter::zip;

/// 合并变换参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MergeArg {
    /// 合并的起点。
    pub start: usize,
    /// 合并的宽度
    pub len: usize,
    /// 分块的顺序。
    pub endian: Option<Endian>,
}

impl<const N: usize> ArrayLayout<N> {
    /// 合并变换是将多个连续维度划分合并的变换。
    /// 大端合并对维度从后到前依次合并。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).merge_be(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_be(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::BigEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 小端合并对维度从前到后依次合并。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0).merge_le(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_le(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: Some(Endian::LittleEndian),
        }])
    }

    /// 合并变换是将多个连续维度划分合并的变换。
    /// 任意合并只考虑维度的存储连续性。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[3, 2, 4], &[4, 12, 1], 0).merge_free(0, 3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge_free(&self, start: usize, len: usize) -> Option<Self> {
        self.merge_many(&[MergeArg {
            start,
            len,
            endian: None,
        }])
    }

    /// 一次对多个阶进行合并变换。
    pub fn merge_many(&self, args: &[MergeArg]) -> Option<Self> {
        let content = self.content();
        let shape = content.shape();
        let strides = content.strides();

        let merged = args.iter().map(|arg| arg.len.max(1)).sum::<usize>();
        if merged == args.len() {
            return Some(self.clone());
        }

        let mut ans = Self::with_ndim(self.ndim + args.len() - merged);

        let mut content = ans.content_mut();
        content.set_offset(self.offset());
        let mut i = 0;
        let mut push = |d, s| {
            content.set_shape(i, d);
            content.set_stride(i, s);
            i += 1;
        };

        let mut last_end = 0;
        for arg in args {
            let &MergeArg { start, len, endian } = arg;
            let end = start + len;

            if len < 2 {
                continue;
            }

            for j in last_end..arg.start {
                push(shape[j], strides[j]);
            }

            let mut pairs = Vec::with_capacity(len);
            for (&d, &s) in zip(&shape[start..end], &strides[start..end]) {
                match d {
                    0 => todo!(),
                    1 => {}
                    _ => pairs.push((d, s)),
                }
            }

            last_end = end;

            if pairs.is_empty() {
                push(1, 0);
                continue;
            }
            match endian {
                Some(Endian::BigEndian) => pairs.reverse(),
                Some(Endian::LittleEndian) => {}
                None => pairs.sort_unstable_by_key(|(_, s)| s.unsigned_abs()),
            }

            let ((d, s), pairs) = pairs.split_first().unwrap();
            let mut d = *d;

            for &(d_, s_) in pairs {
                if s_ == s * d as isize {
                    d *= d_
                } else {
                    return None;
                }
            }

            push(d, *s);
        }
        for j in last_end..shape.len() {
            push(shape[j], strides[j]);
        }

        Some(ans)
    }
}

#[test]
fn test_merge_return_none() {
    let layout = ArrayLayout::<3>::new(&[16, 4, 2], &[8, 4, 1], 0).merge_be(0, 3);
    assert!(layout.is_none());
}

#[test]
fn test_merge_pairs_empyt() {
    let layout = ArrayLayout::<3>::new(&[1, 1, 1], &[1, 1, 1], 0)
        .merge_be(0, 2)
        .unwrap();
    assert_eq!(layout.shape(), &[1, 1]);
    assert_eq!(layout.strides(), &[0, 1]);
    assert_eq!(layout.offset(), 0);
}

#[test]
fn test_merge_be_example() {
    let layout = ArrayLayout::<3>::new(&[16, 1, 4], &[16, 768, 4], 0)
        .merge_be(0, 2)
        .unwrap();
    assert_eq!(layout.shape(), &[16, 4]);
    assert_eq!(layout.strides(), &[16, 4]);
    assert_eq!(layout.offset(), 0);
}

#[test]
fn test_merge_le_example() {
    let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0);
    let merged_layout = layout.merge_le(0, 3).unwrap();

    assert_eq!(merged_layout.shape(), &[24]);
    assert_eq!(merged_layout.strides(), &[1]);
    assert_eq!(merged_layout.offset(), 0);
}

#[test]
fn test_merge_len_zero() {
    let layout = ArrayLayout::<3>::new(&[4, 3, 2], &[1, 4, 12], 0);
    let merged_layout = layout.merge_le(0, 0).unwrap();

    assert_eq!(merged_layout.shape(), &[4, 3, 2]);
    assert_eq!(merged_layout.strides(), &[1, 4, 12]);
    assert_eq!(merged_layout.offset(), 0);
}

#[test]
fn test_partial_merge() {
    let layout = ArrayLayout::<4>::new(&[2, 3, 4, 5], &[60, 20, 5, 1], 0);
    let merged_layout = layout.merge_be(1, 2).unwrap();

    assert_eq!(merged_layout.shape(), &[2, 12, 5]);
    assert_eq!(merged_layout.strides(), &[60, 5, 1]);
    assert_eq!(merged_layout.offset(), 0);
}

#[test]
fn test_merge_free_example() {
    let layout = ArrayLayout::<3>::new(&[3, 2, 4], &[4, 12, 1], 0);
    let merged_layout = layout.merge_free(0, 3).unwrap();

    assert_eq!(merged_layout.shape(), &[24]);
    assert_eq!(merged_layout.strides(), &[1]);
    assert_eq!(merged_layout.offset(), 0);
}
