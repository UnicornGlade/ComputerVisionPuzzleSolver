#include "gaussian_blur.h"

#include <cmath>
#include <vector>

#include <libbase/runtime_assert.h>

namespace libimages {

static std::vector<float> gaussian_kernel_1d(float sigma, int* out_radius) {
    rassert(sigma > 0.0f, "sigma must be positive", sigma);

    const int radius = static_cast<int>(std::ceil(3.0f * sigma));
    *out_radius = radius;

    std::vector<float> k(static_cast<std::size_t>(2 * radius + 1));
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    float sum = 0.0f;
    for (int x = -radius; x <= radius; ++x) {
        const float v = std::exp(-static_cast<float>(x * x) * inv2s2);
        k[static_cast<std::size_t>(x + radius)] = v;
        sum += v;
    }
    for (float& v : k) v /= sum;
    return k;
}

image32f gaussian_blur_gray(const image32f& gray, float sigma) {
    rassert(gray.channels() == 1, "gaussian_blur_gray expects grayscale", gray.channels());

    const int w = gray.width();
    const int h = gray.height();

    int radius = 0;
    const std::vector<float> k = gaussian_kernel_1d(sigma, &radius);

    image32f tmp(w, h, 1);
    image32f out(w, h, 1);

    // Horizontal pass (replicate border).
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            float acc = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx) {
                int x = i + dx;
                if (x < 0) x = 0;
                if (x >= w) x = w - 1;
                acc += gray(j, x) * k[static_cast<std::size_t>(dx + radius)];
            }
            tmp(j, i) = acc;
        }
    }

    // Vertical pass (replicate border).
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            float acc = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                int y = j + dy;
                if (y < 0) y = 0;
                if (y >= h) y = h - 1;
                acc += tmp(y, i) * k[static_cast<std::size_t>(dy + radius)];
            }
            out(j, i) = acc;
        }
    }

    return out;
}

} // namespace libimages
