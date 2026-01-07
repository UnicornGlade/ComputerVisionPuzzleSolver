#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <libbase/point2.h>
#include <libimages/image.h>

namespace find_mask_borders {

// Simple RGB color holder.
struct point3u final {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

// 1) Border pixels mask:
// input: 1-channel mask {0,255}
// output: 1-channel mask {0,255}, where 255 are object pixels having >=min_bg_neighbors background neighbors in 8-neighborhood.
// Neighbors outside image are treated as background.
libimages::image8u border_pixels_mask(const libimages::image8u& object_mask, int min_bg_neighbors = 2);

// 2) Split border pixels into `side_count` groups ("sides") using DSU + Sobel gradient angles:
// - compute Sobel gradients of object_mask (0 outside, 255 inside) => gradients point inward.
// - build list of adjacent border pixel pairs (8-neigh edges).
// - sort edges by angle difference of gradients (small first).
// - DSU-unite edges until number of sets <= side_count (or no more edges).
// Returns: vector of sides, each side is vector of pixels ordered approximately clockwise along that side.
std::vector<std::vector<point2i>> split_border_into_sides(const libimages::image8u& object_mask,
                                                         const libimages::image8u& border_mask,
                                                         int side_count = 4);

// Debug for sampling.
struct SamplingDebugParams {
    std::filesystem::path out_dir;
    std::string prefix;          // e.g. "0000"
    std::string dump_ext = ".png";
    bool force_overwrite = true;
    bool verbose = false;

    int point_radius = 1;        // radius in pixels for visualizing points
};

// 3) Sample colors along each side line (from first pixel to last pixel):
// - places L sampling points uniformly along the segment [first,last] (including endpoints if L>=2).
// - for each sample point, finds two donor pixels on the side that bracket it by projection,
//   then linearly interpolates their colors.
// Returns: K vectors (K = sides.size()), each has L colors.
std::vector<std::vector<point3u>> sample_sides_colors(const libimages::image8u& image_rgb_or_gray,
                                                      const std::vector<std::vector<point2i>>& sides_clockwise,
                                                      int samples_per_side = 10,
                                                      SamplingDebugParams* dbg = nullptr);

// Helpers for visualization / output assembly.
libimages::image8u visualize_sides_overlay(const libimages::image8u& image_rgb_or_gray,
                                          const std::vector<std::vector<point2i>>& sides,
                                          std::uint32_t seed = 123,
                                          std::uint8_t darken_div = 2);

libimages::image8u make_kxl_image(const std::vector<std::vector<point3u>>& samples);

} // namespace find_mask_borders
