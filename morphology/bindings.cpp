#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "morphology.h"

namespace py = pybind11;

PYBIND11_MODULE(morphology_cpp, m) {
    py::enum_<MorphologyOp>(m, "MorphologyOp")
        .value("ERODE", MorphologyOp::ERODE)
        .value("DILATE", MorphologyOp::DILATE)
        .value("OPEN", MorphologyOp::OPEN)
        .value("CLOSE", MorphologyOp::CLOSE)
        .export_values();

    m.def("morphology_operation", [](py::array_t<uint8_t> image, int width, int height, MorphologyOp op, int kernel_size) -> py::array_t<uint8_t> {
        py::buffer_info buf = image.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1D array");
        }
        
        std::vector<uint8_t> image_vec(static_cast<uint8_t*>(buf.ptr), static_cast<uint8_t*>(buf.ptr) + buf.size);
        
        auto result = morphology_operation(image_vec, width, height, op, kernel_size);
        
        py::array_t<uint8_t> output(result.size());
        py::buffer_info out_buf = output.request();
        std::copy(result.begin(), result.end(), static_cast<uint8_t*>(out_buf.ptr));
        
        return output;
    }, py::arg("image"), py::arg("width"), py::arg("height"), py::arg("op"), py::arg("kernel_size") = 3);
}
