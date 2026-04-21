#include "morphology.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>

static std::vector<double> create_gaussian_kernel(int kernel_size, double sigma) {
    std::vector<double> kernel(kernel_size * kernel_size);
    int half = kernel_size / 2;
    double sum = 0.0;
    double two_sigma_sq = 2.0 * sigma * sigma;
    
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            double val = std::exp(-(x * x + y * y) / two_sigma_sq);
            kernel[(y + half) * kernel_size + (x + half)] = val;
            sum += val;
        }
    }
    
    for (auto& val : kernel) {
        val /= sum;
    }
    
    return kernel;
}

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

std::vector<uint8_t> gaussian_filter(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size,
    double sigma
) {
    std::vector<uint8_t> result(width * height, 0);
    std::vector<double> kernel = create_gaussian_kernel(kernel_size, sigma);
    int half = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        uint8_t val = image[py * width + px];
                        double k = kernel[(ky + half) * kernel_size + (kx + half)];
                        sum += static_cast<double>(val) * k;
                    }
                }
            }
            
            result[y * width + x] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, sum)));
        }
    }
    
    return result;
}

std::vector<uint8_t> gaussian_filter_rgb(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int kernel_size,
    double sigma
) {
    std::vector<uint8_t> result(width * height * 3);
    std::vector<double> kernel = create_gaussian_kernel(kernel_size, sigma);
    int half = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        int idx = (py * width + px) * 3;
                        uint8_t r = image[idx];
                        uint8_t g = image[idx + 1];
                        uint8_t b = image[idx + 2];
                        double k = kernel[(ky + half) * kernel_size + (kx + half)];
                        sum_r += static_cast<double>(r) * k;
                        sum_g += static_cast<double>(g) * k;
                        sum_b += static_cast<double>(b) * k;
                    }
                }
            }
            
            int out_idx = (y * width + x) * 3;
            result[out_idx] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, sum_r)));
            result[out_idx + 1] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, sum_g)));
            result[out_idx + 2] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, sum_b)));
        }
    }
    
    return result;
}
