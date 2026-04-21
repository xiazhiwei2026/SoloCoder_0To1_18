#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <vector>
#include <cstdint>

enum class MorphologyOp {
    ERODE,
    DILATE,
    OPEN,
    CLOSE
};

std::vector<uint8_t> morphology_operation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    MorphologyOp op,
    int kernel_size = 3
);

std::vector<uint8_t> gaussian_filter(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size = 3,
    double sigma = 1.0
);

std::vector<uint8_t> gaussian_filter_rgb(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size = 3,
    double sigma = 1.0
);

#endif
