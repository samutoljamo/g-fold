#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "lib.cpp"

namespace py = pybind11;
PYBIND11_MODULE(gfold, m) {
    m.doc() = "G-FOLD";
    py::class_<Result>(m, "Result")
            .def(py::init<>())
            .def_readwrite("z", &Result::z)
            .def_readwrite("s", &Result::s)
            .def_readwrite("x", &Result::x)
            .def_readwrite("u", &Result::u)
            .def_readwrite("status", &Result::status)
            .def_readwrite("t", &Result::t);
    py::class_<Vector>(m, "Vector")
            .def(py::init<>())
            .def_readwrite("x", &Vector::x)
            .def_readwrite("y", &Vector::y)
            .def_readwrite("z", &Vector::z);
    py::class_<Spacecraft>(m, "Spacecraft")
            .def(py::init<>())
            .def_readwrite("mass", &Spacecraft::mass)
            .def_readwrite("fuel", &Spacecraft::fuel)
            .def_readwrite("a", &Spacecraft::a)
            .def_readwrite("min_thrust", &Spacecraft::min_thrust)
            .def_readwrite("max_thrust", &Spacecraft::max_thrust)
            .def_readwrite("max_angle", &Spacecraft::max_angle)
            .def_readwrite("max_vel", &Spacecraft::max_vel)
            .def_readwrite("glide_slope_angle", &Spacecraft::glide_slope_angle)
            .def_readwrite("initial_position", &Spacecraft::initial_position)
            .def_readwrite("initial_velocity", &Spacecraft::initial_velocity)
            .def_readwrite("target_velocity", &Spacecraft::target_velocity);
    py::class_<Settings>(m, "Settings")
            .def(py::init<>())
            .def_readwrite("gravity", &Settings::gravity)
            .def_readwrite("maxit", &Settings::maxit);
    py::class_<Problem>(m, "Problem")
            .def(py::init<Settings, Spacecraft>())
            .def("solve_at", &Problem::solve_at)
            .def("solve", &Problem::solve);
}
