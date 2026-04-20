#include "morphology.h"
#include <algorithm>
#include <vector>

static std::vector<uint8_t> erode(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size
) {
    std::vector<uint8_t> result(width * height, 255);
    int half = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t min_val = 255;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        uint8_t val = image[py * width + px];
                        min_val = std::min(min_val, val);
                    }
                }
            }
            
            result[y * width + x] = min_val;
        }
    }
    
    return result;
}

static std::vector<uint8_t> dilate(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size
) {
    std::vector<uint8_t> result(width * height, 0);
    int half = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t max_val = 0;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        uint8_t val = image[py * width + px];
                        max_val = std::max(max_val, val);
                    }
                }
            }
            
            result[y * width + x] = max_val;
        }
    }
    
    return result;
}

std::vector<uint8_t> morphology_operation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    MorphologyOp op,
    int kernel_size
) {
    switch (op) {
        case MorphologyOp::ERODE:
            return erode(image, width, height, kernel_size);
        case MorphologyOp::DILATE:
            return dilate(image, width, height, kernel_size);
        case MorphologyOp::OPEN: {
            auto temp = erode(image, width, height, kernel_size);
            return dilate(temp, width, height, kernel_size);
        }
        case MorphologyOp::CLOSE: {
            auto temp = dilate(image, width, height, kernel_size);
            return erode(temp, width, height, kernel_size);
        }
        default:
            return image;
    }
}
