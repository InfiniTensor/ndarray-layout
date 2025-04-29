# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-04-29

### Fixed

- Compatible with meaningless input: Merging 0 or 1 dimensions will not change the layout;

### Added

- Add `to_inline_size` function, to copy data from `ArrayLayout<N>` into `ArrayLayout<M>`;
- Complete documentation and testing;

## [0.2.1] - 2025-03-28

### Added

- Add `write_array` to write Nd array to `fmt::Formatter`;

## [0.2.0] - 2025-02-23

### Added

- Add `num_elements` to calculate the number of elements in array;
- Add `element_offset` to calculate the offset of element at the given index;

### Changed

- Upgrade Rust to 2024 edition;

### Fixed

- Fix merging dims with length 1;

[Unreleased]: https://github.com/InfiniTensor/ndarray-layout/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/InfiniTensor/ndarray-layout/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/InfiniTensor/ndarray-layout/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/InfiniTensor/ndarray-layout/releases/tag/v0.2.0
