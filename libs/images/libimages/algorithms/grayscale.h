#pragma once

#include <libimages/image.h>

namespace libimages {

image32f to_grayscale_float(const image8u& img);

} // namespace libimages