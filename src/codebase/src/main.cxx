#include <pybind11/pybind11.h>
#include "alpha/alpha.h"

PYBIND11_MODULE(main, m)
{
    pybind11::class_<Alpha>(m, "main")
        .def(pybind11::init<>())
        .def("start", &Alpha::main);
}
