#pragma once

#include <cstdint>

#include <libimages/image.h>

namespace libimages {

struct Gradients {
    image32f dx;    // 1-channel
    image32f dy;    // 1-channel
    image32f mag;   // 1-channel
    image32f angle; // 1-channel degrees in [0,360)
};

Gradients sobel_gradients(const image32f& gray_blurred);

// Debug helpers.
image8u visualize_signed_to_u8(const image32f& img);
image8u visualize_angle_hsv(const image32f& angle_deg, const image32f& mag);

float wrap_angle_deg(float a);
float angle_diff_deg(float a, float b);

} // namespace libimages
