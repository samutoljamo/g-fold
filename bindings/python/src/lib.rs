// The #[pymodule] lives in gfold-ffi behind the `python` feature; re-exporting
// its symbol here lets maturin build this crate as the `gfold._gfold` extension.
pub use gfold_ffi::python::*;
