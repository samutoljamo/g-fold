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

#[derive(Debug, Clone)]
pub struct Row {
    pub coeffs: Vec<(usize, f64)>,
    pub b: f64,
}

pub fn eval_row(row: &Row, point: &[f64]) -> f64 {
    row.coeffs.iter().map(|&(idx, c)| c * point[idx]).sum()
}

#[derive(Debug)]
pub struct Builder {
    pub layout: Layout,
    pub rows: Vec<Row>,
}

impl Builder {
    pub fn new(layout: Layout) -> Self { Self { layout, rows: Vec::new() } }
    pub fn push(&mut self, row: Row) { self.rows.push(row); }
    pub fn nrows(&self) -> usize { self.rows.len() }
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

    #[test]
    fn eval_row_dots_point() {
        let l = Layout { n: 2 };
        let row = Row { coeffs: vec![(l.x(0,0), 2.0), (l.z(1), -1.0)], b: 3.0 };
        let mut point = vec![0.0; l.nvars()];
        point[l.x(0,0)] = 5.0;
        point[l.z(1)] = 4.0;
        assert_eq!(eval_row(&row, &point), 2.0 * 5.0 + (-1.0) * 4.0);
    }
}
