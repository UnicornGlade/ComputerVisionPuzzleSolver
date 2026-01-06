#pragma once

#include <libimages/image.h>

namespace libimages {

image8u downscale_nearest(const image8u &img, float downscale_ratio);

image8u downscale_bilinear(const image8u &img, float downscale_ratio);

image8u downscale_area(const image8u &img, float downscale_ratio);

}