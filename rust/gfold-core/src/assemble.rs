//! Problem assembly: variable index map and conic problem construction.

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub n: usize,
}

impl Layout {
    pub fn nvars(&self) -> usize { 11 * self.n }
    pub fn x(&self, i: usize, comp: usize) -> usize { 6 * i + comp }
    pub fn u(&self, i: usize, comp: usize) -> usize { 6 * self.n + 3 * i + comp }
    pub fn s(&self, i: usize) -> usize { 9 * self.n + i }
    pub fn z(&self, i: usize) -> usize { 10 * self.n + i }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_offsets() {
        let l = Layout { n: 100 };
        assert_eq!(l.nvars(), 1100);
        assert_eq!(l.x(0, 0), 0);
        assert_eq!(l.x(0, 5), 5);
        assert_eq!(l.x(1, 0), 6);
        assert_eq!(l.u(0, 0), 600);
        assert_eq!(l.u(1, 2), 605);
        assert_eq!(l.s(0), 900);
        assert_eq!(l.s(99), 999);
        assert_eq!(l.z(0), 1000);
        assert_eq!(l.z(99), 1099);
    }
}
