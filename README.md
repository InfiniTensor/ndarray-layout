# ndarray-layout

[![CI](https://github.com/InfiniTensor/ndarray-layout/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/ndarray-layout/actions)
[![Latest version](https://img.shields.io/crates/v/ndarray-layout.svg)](https://crates.io/crates/ndarray-layout)
[![Documentation](https://docs.rs/ndarray-layout/badge.svg)](https://docs.rs/ndarray-layout)
[![license](https://img.shields.io/github/license/InfiniTensor/ndarray-layout)](https://mit-license.org/)
[![codecov](https://codecov.io/github/Simon25772/ndarray-layout/branch/ShenghuSu/graph/badge.svg)](https://codecov.io/github/Simon25772/ndarray-layout/tree/Shenghu)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/ndarray-layout)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/ndarray-layout)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/ndarray-layout)](https://github.com/InfiniTensor/ndarray-layout/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/ndarray-layout)](https://github.com/InfiniTensor/ndarray-layout/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/ndarray-layout)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/ndarray-layout)

*ndarray-layout* 是一个用于处理多维数组布局的 *Rust* 库，它提供了 `ArrayLayout` 结构体，用于高效管理和操作多维数组的元信息，如形状、步长和偏移量等。这个库在处理多维数组时，提供了灵活且高效的布局管理方式，能够满足不同场景下对数组布局的操作需求。

## 主要功能特点

### 多维数组布局管理

- `ArrayLayout` 结构体支持指定任意维度的数组布局，通过 `new` 方法可以创建具有指定形状、步长和偏移量的布局;
- 提供 `new_contiguous` 方法，用于创建连续的数组布局，支持大端序（`BigEndian`）和小端序（`LittleEndian`）两种存储顺序;

### 元信息访问

- 提供便捷的方法来访问数组布局的元信息，如 `ndim`、`offset`、`shape` 和 `strides` 等;
- 支持计算数组元素的偏移量和数据范围，方便进行内存访问和数据处理;

### 布局操作功能

- 提供多种布局变换方法，如 `index`、`tile`、`transpose`、`merge` 和 `slice` 等，方便对数组布局进行各种变换操作;

## 使用示例

```rust
use ndarray_layout::{ArrayLayout, BroadcastArg};

// 创建一个新的 `ArrayLayout` 实例。
// 形状为 [1, 2, 3]，步长为 [12, 4, 1]，偏移量为 0。
let layout = ArrayLayout::<3>::new(&[1, 2, 3], &[12, 4, 1], 0);

// 验证初始的形状和步长。
assert_eq!(layout.shape(), &[1, 2, 3]);
assert_eq!(layout.strides(), &[12, 4, 1]);
assert_eq!(layout.offset(), 0);

// 对第 0 维进行广播变换，广播次数为 4。
let broadcasted_layout = layout.broadcast(0, 4);

// 验证广播变换后的形状和步长。
assert_eq!(broadcasted_layout.shape(), &[4, 2, 3]);
assert_eq!(broadcasted_layout.strides(), &[0, 4, 1]);
assert_eq!(broadcasted_layout.offset(), 0);
```
