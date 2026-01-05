#pragma once

#include <libimages/image.h>

namespace libimages {

image8u downscale_nearest(const image8u &img, float downscale_ratio);

}