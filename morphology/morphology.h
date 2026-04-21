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

#endif
