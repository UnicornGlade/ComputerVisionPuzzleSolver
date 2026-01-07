#pragma once

#include <libimages/algorithms/sobel_gradients.h>
#include <libimages/image.h>

namespace libimages {

// Returns 1-channel mask:
// 1 if pixel is a local maximum of gradient magnitude along the gradient direction,
// 0 otherwise (if magnitude is <= at least one of the two selected neighbors).
image8u non_maximum_suppression(const Gradients& g);

// Zeroes gradients at pixels where is_ok == 0.
// is_ok must be 1-channel, same width/height.
void apply_mask_inplace(Gradients& g, const image8u& is_ok);

// Builds a 1-channel mask: 1 if magnitude >= threshold, else 0.
image8u magnitude_threshold_mask(const Gradients& g, float magnitude_threshold);

} // namespace libimages
