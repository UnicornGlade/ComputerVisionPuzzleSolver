#pragma once

#include <string>

#include <libimages/image.h>

namespace libimages {

image8u load_image(const std::string &path);

// Saves 1/3/4-channel 8-bit image. For JPEG, alpha is dropped.
void save_image(const image8u &img, const std::string &path, int jpg_quality = 95);

} // namespace libimages
