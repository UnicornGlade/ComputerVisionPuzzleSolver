// find_mask_borders.h

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <libbase/point2.h>
#include <libimages/image.h>

namespace find_mask_borders {

struct point3u final {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

libimages::image8u border_pixels_mask(const libimages::image8u& object_mask, int min_bg_neighbors = 2);

// Debug for split_border_into_sides (padded intermediate dumps).
struct SplitSidesDebugParams {
    std::filesystem::path out_dir;
    std::string prefix;              // e.g. "0000_"
    std::string dump_ext = ".png";
    bool force_overwrite = true;
    bool verbose = false;

    int pad = 10;                    // internal padding used for Sobel prep
    float blur_sigma = 0.25f;         // blur before Sobel (on padded mask)
    int arrow_stride = 6;            // draw arrow every N pixels
    float arrow_min_mag = 5.0f;      // skip arrows with mag below this
    float arrow_len_px = 6.0f;       // arrow length in pixels
    int arrow_thickness = 1;         // arrow line thickness in pixels
};

// CHANGED: optional dbg at the end (default nullptr), keeps old calls intact.
std::vector<std::vector<point2i>> split_border_into_sides(const libimages::image8u& object_mask,
                                                         const libimages::image8u& border_mask,
                                                         int side_count = 4,
                                                         SplitSidesDebugParams* dbg = nullptr);

struct SamplingDebugParams {
    std::filesystem::path out_dir;
    std::string prefix;
    std::string dump_ext = ".png";
    bool force_overwrite = true;
    bool verbose = false;
    int point_radius = 1;
};

std::vector<std::vector<point3u>> sample_sides_colors(const libimages::image8u& image_rgb_or_gray,
                                                      const std::vector<std::vector<point2i>>& sides_clockwise,
                                                      int samples_per_side = 10,
                                                      SamplingDebugParams* dbg = nullptr);

libimages::image8u visualize_sides_overlay(const libimages::image8u& image_rgb_or_gray,
                                          const std::vector<std::vector<point2i>>& sides,
                                          std::uint32_t seed = 123,
                                          std::uint8_t darken_div = 2);

libimages::image8u make_kxl_image(const std::vector<std::vector<point3u>>& samples);

} // namespace find_mask_borders
