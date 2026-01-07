#pragma once

#include <cstdint>

#include <libimages/image.h>

namespace background_masking {

struct Params {
    // Morphology post-processing strengths (square structuring element radius).
    // Set to 0 to disable that step.
    int dilate_strength = 2; // bigger -> fills gaps / connects blobs more
    int erode_strength = 2;  // bigger -> removes thin protrusions / shrinks more (after dilation: closing)
};

// 1) Collect perimeter pixel values
// 2) Print stats (summaryStats + percentile(90))
// 3) threshold = 1.5 * p90
// Returned value is clamped to [0, 255].
float estimate_background_threshold(const libimages::image8u& gray);

// Build foreground mask:
// - 255 where gray > threshold
// - 0 otherwise
// Then apply morphology: dilate (if >0) then erode (if >0).
libimages::image8u build_foreground_mask(const libimages::image8u& gray, const Params& p = Params{});

} // namespace background_masking
