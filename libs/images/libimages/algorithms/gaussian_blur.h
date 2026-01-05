#pragma once

#include <libimages/image.h>

namespace libimages {

image32f gaussian_blur_gray(const image32f& gray, float sigma);

} // namespace libimages