//! Typed Python bindings. Config/nested structs come from gfold-core (pyclass
//! under its `python` feature); Trajectory is wrapped here to return NumPy arrays.
use gfold_core::config::{Config, Environment, Solver, Spacecraft};
use gfold_core::solve::{solve as core_solve, Trajectory};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
struct PyTrajectory {
    inner: Trajectory,
}

fn nx3(py: Python<'_>, rows: &[[f64; 3]]) -> Py<PyArray2<f64>> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), 3), flat)
        .expect("n*3")
        .into_pyarray(py)
        .unbind()
}

fn vec1(py: Python<'_>, v: &[f64]) -> Py<PyArray1<f64>> {
    v.to_vec().into_pyarray(py).unbind()
}

#[pymethods]
impl PyTrajectory {
    #[getter]
    fn positions(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        nx3(py, &self.inner.positions)
    }
    #[getter]
    fn velocities(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        nx3(py, &self.inner.velocities)
    }
    #[getter]
    fn u_values(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        nx3(py, &self.inner.u_values)
    }
    #[getter]
    fn thrusts(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vec1(py, &self.inner.thrusts)
    }
    #[getter]
    fn normalized_thrusts(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vec1(py, &self.inner.normalized_thrusts)
    }
    #[getter]
    fn z_values(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vec1(py, &self.inner.z_values)
    }
    #[getter]
    fn s_values(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vec1(py, &self.inner.s_values)
    }
    #[getter]
    fn time_points(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vec1(py, &self.inner.time_points)
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective
    }
    #[getter]
    fn final_mass(&self) -> f64 {
        self.inner.final_mass
    }
    #[getter]
    fn status(&self) -> String {
        self.inner.status.clone()
    }
}

#[pyfunction]
fn solve(config: PyRef<'_, Config>) -> PyResult<PyTrajectory> {
    core_solve(&*config)
        .map(|inner| PyTrajectory { inner })
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn _gfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add_class::<Spacecraft>()?;
    m.add_class::<Environment>()?;
    m.add_class::<Solver>()?;
    m.add_class::<PyTrajectory>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
