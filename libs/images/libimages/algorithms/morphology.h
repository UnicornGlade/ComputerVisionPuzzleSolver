#pragma once

#include <cstdint>

#include <libimages/image.h>

namespace morphology {

// Binary morphology on 1-channel image8u (pixels must be 0 or 255).
// strength = radius of a square structuring element (Chebyshev distance).
// Border handling: zero-padding outside the image.
//
// strength == 0 -> returns a copy.

libimages::image8u erode(const libimages::image8u& src, int strength);
libimages::image8u dilate(const libimages::image8u& src, int strength);

} // namespace morphology
