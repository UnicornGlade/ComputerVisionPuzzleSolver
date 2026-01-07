#pragma once

#include <cstdint>
#include <vector>

#include <libbase/bbox2.h>
#include <libbase/algorithms/disjoint_set.h>
#include <libbase/point2.h>
#include <libbase/runtime_assert.h>

#include <libimages/image.h>

namespace utils {

struct ExtractParams {
    bool eight_connected = true; // true -> 8-neighborhood, false -> 4-neighborhood
};

// One extracted connected component (foreground object).
template <typename T> struct ExtractedObject {
    libimages::Image<T> image; // cropped image (same channels as input)
    libimages::image8u mask;   // cropped mask (1 channel, 0/255)
    point2i offset;            // top-left in the original image that maps to (0,0)
    bbox2i bbox;               // bbox in original coordinates (half-open)
};

// Extract connected components of mask==255 using DSU.
// - input: any libimages::Image<T>
// - mask: same W/H, 1-channel, values 0 or 255
// Returns vector of cropped (image,mask,offset) for each component.
template <typename T>
std::vector<ExtractedObject<T>> extract_objects_by_mask(const libimages::Image<T>& img,
                                                        const libimages::image8u& mask,
                                                        const ExtractParams& p = {});

// Explicit instantiations provided in utils.cpp:
extern template std::vector<ExtractedObject<std::uint8_t>>
extract_objects_by_mask<std::uint8_t>(const libimages::Image<std::uint8_t>&,
                                      const libimages::image8u&,
                                      const ExtractParams&);

extern template std::vector<ExtractedObject<float>>
extract_objects_by_mask<float>(const libimages::Image<float>&,
                               const libimages::image8u&,
                               const ExtractParams&);

} // namespace libimages::utils
