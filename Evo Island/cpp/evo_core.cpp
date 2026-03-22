#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "simulation_core.h"

namespace py = pybind11;

PYBIND11_MODULE(evo_core, m) {
    m.doc() = "C++ simulation core for Morphanimals / Evo Island";

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("extinct",                      &StepResult::extinct)
        .def_readonly("population_count",             &StepResult::population_count)
        .def_readonly("species_counts",               &StepResult::species_counts)
        .def_readonly("deaths_aging",                 &StepResult::deaths_aging)
        .def_readonly("deaths_competition",           &StepResult::deaths_competition)
        .def_readonly("deaths_starvation",            &StepResult::deaths_starvation)
        .def_readonly("deaths_exposure",              &StepResult::deaths_exposure)
        .def_readonly("deaths_predation",             &StepResult::deaths_predation)
        .def_readonly("total_age",                    &StepResult::total_age)
        .def_readonly("total_lifespan",               &StepResult::total_lifespan)
        .def_readonly("total_strength",               &StepResult::total_strength)
        .def_readonly("total_hardiness",              &StepResult::total_hardiness)
        .def_readonly("total_metabolism",             &StepResult::total_metabolism)
        .def_readonly("total_reproduction_threshold", &StepResult::total_reproduction_threshold)
        .def_readonly("total_speed",                  &StepResult::total_speed)
        .def_readonly("total_trophism",               &StepResult::total_trophism)
        .def_readonly("total_kin_attraction",          &StepResult::total_kin_attraction)
        .def_readonly("total_threat_response",        &StepResult::total_threat_response);

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<py::dict, py::list, py::list, int, int>(),
             py::arg("cfg"),
             py::arg("world_matrix"),
             py::arg("food_matrix"),
             py::arg("start_i"),
             py::arg("start_j"))
        .def("step",                 &Simulation::step,
             py::arg("current_step"),
             "Run one simulation step and return a StepResult.")
        .def("get_attribute_matrix", &Simulation::get_attribute_matrix,
             py::arg("attr"),
             "Return an (N,N) or (N,N,3) numpy array for a named agent attribute.")
        .def("is_extinct",           &Simulation::is_extinct);
}
