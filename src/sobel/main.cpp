#include <cmath>
#include <iostream>
#include <string>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

static libimages::image32f to_grayscale_float(const libimages::image8u &img) {
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported channel count",
            img.channels());

    libimages::image32f gray(img.width(), img.height(), 1);

    if (img.channels() == 1) {
        for (int j = 0; j < img.height(); ++j) {
            for (int i = 0; i < img.width(); ++i) {
                gray(j, i) = static_cast<float>(img(j, i));
            }
        }
        return gray;
    }

    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            const float r = static_cast<float>(img(j, i, 0));
            const float g = static_cast<float>(img(j, i, 1));
            const float b = static_cast<float>(img(j, i, 2));
            gray(j, i) = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
    return gray;
}

static libimages::image32f sobel_magnitude_l2(const libimages::image32f &gray) {
    rassert(gray.channels() == 1, "sobel_magnitude_l2 expects grayscale", gray.channels());

    const int w = gray.width();
    const int h = gray.height();

    libimages::image32f mag(w, h, 1);
    mag.fill(0.0f);

    if (w < 3 || h < 3)
        return mag;

    for (int j = 1; j < h - 1; ++j) {
        for (int i = 1; i < w - 1; ++i) {
            const float p00 = gray(j - 1, i - 1);
            const float p01 = gray(j - 1, i);
            const float p02 = gray(j - 1, i + 1);

            const float p10 = gray(j, i - 1);
            const float p11 = gray(j, i);
            const float p12 = gray(j, i + 1);

            const float p20 = gray(j + 1, i - 1);
            const float p21 = gray(j + 1, i);
            const float p22 = gray(j + 1, i + 1);

            const float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            const float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

            mag(j, i) = std::sqrt(gx * gx + gy * gy);
        }
    }

    return mag;
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: sobel_app <input_image.(png|jpg|jpeg)> <output_image.(png|jpg|jpeg)>\n";
            return 1;
        }

        const std::string input_path = argv[1];
        const std::string output_path = argv[2];

        const libimages::image8u input = libimages::load_image(input_path);
        const libimages::image32f gray = to_grayscale_float(input);
        const libimages::image32f mag = sobel_magnitude_l2(gray);

        // Debug-style visualization: auto-scale by max, save as 8-bit image
        libimages::debug_io::dump_image(output_path, mag, /*verbose=*/true, /*force=*/true);

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
