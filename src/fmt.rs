use crate::ArrayLayout;
use std::fmt;

impl<const N: usize> ArrayLayout<N> {
    /// 高维数组格式化。
    ///
    /// # Safety
    ///
    /// 这个函数从对裸指针解引用以获得要格式化的数组元素。
    pub unsafe fn write_array<T: fmt::Display + Copy>(
        &self,
        f: &mut fmt::Formatter,
        ptr: *const T,
    ) -> fmt::Result {
        match self.ndim() {
            0 => {
                write!(f, "array<> = [{}]", unsafe {
                    ptr.byte_offset(self.offset()).read_unaligned()
                })
            }
            1 => {
                let &[n] = self.shape() else { unreachable!() };
                let &[s] = self.strides() else { unreachable!() };

                writeln!(f, "array<{n}>[")?;
                let ptr = unsafe { ptr.byte_offset(self.offset()) };
                for i in 0..n as isize {
                    writeln!(f, "    {}", unsafe {
                        ptr.byte_offset(i * s).read_unaligned()
                    })?
                }
                writeln!(f, "]")?;
                Ok(())
            }
            _ => {
                let mut title = "array<".to_string();
                for d in self.shape() {
                    title.push_str(&format!("{d}x"))
                }
                assert_eq!(title.pop(), Some('x'));
                title.push('>');

                let mut stack = Vec::with_capacity(self.ndim() - 2);
                self.write_recursive(f, ptr, &title, &mut stack)
            }
        }
    }

    fn write_recursive<T: fmt::Display>(
        &self,
        f: &mut fmt::Formatter,
        ptr: *const T,
        title: &str,
        indices: &mut Vec<usize>,
    ) -> fmt::Result {
        match *self.shape() {
            [] | [_] => unreachable!(),
            [rows, cols] => {
                write!(f, "{title}[")?;
                for i in indices {
                    write!(f, "{i}, ")?
                }
                writeln!(f, "..]")?;

                let &[rs, cs] = self.strides() else {
                    unreachable!()
                };

                let ptr = unsafe { ptr.byte_offset(self.offset()) };
                for r in 0..rows as isize {
                    for c in 0..cols as isize {
                        write!(f, "{} ", unsafe {
                            ptr.byte_offset(r * rs + c * cs).read_unaligned()
                        })?
                    }
                    writeln!(f)?
                }
            }
            [batch, ..] => {
                for i in 0..batch {
                    indices.push(i);
                    self.index(0, i).write_recursive(f, ptr, title, indices)?;
                    indices.pop();
                }
            }
        }
        Ok(())
    }
}

#[test]
fn test() {
    const DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0];

    struct Tensor(ArrayLayout<4>);

    impl fmt::Display for Tensor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe { self.0.write_array(f, DATA.as_ptr()) }
        }
    }

    let tensor = Tensor(ArrayLayout::<4>::new_contiguous(
        &[DATA.len()],
        crate::Endian::BigEndian,
        1,
    ));
    println!("{}", tensor);

    let tensor = Tensor(tensor.0.tile_be(0, &[1, DATA.len()]).broadcast(0, 6));
    println!("{}", tensor);

    let tensor = Tensor(tensor.0.tile_be(0, &[2, 3]).tile_be(2, &[5, 2]));
    println!("{}", tensor);

    let tensor = Tensor(ArrayLayout::<4>::with_ndim(0));
    println!("{}", tensor);
}
