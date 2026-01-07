#pragma once

#include <cstdint>
#include <vector>

#include <libbase/point2.h>
#include <libimages/image.h>

namespace draw {

struct DrawParams {
    std::uint32_t seed = 123;

    int line_thickness = 2;
    int start_circle_radius = 4;
    int start_circle_thickness = 1;

    int nn_line_thickness = 1;
};

// High-level overlay: darkened input + random-colored pixels per SegmentPixels.
libimages::image8u overlay_segments_pixels(const libimages::image8u& input,
                                          const std::vector<std::vector<point2i>>& segments,
                                          std::uint32_t seed);

// High-level overlay: darkened input + random-colored segments (A->B) + circle at A.
libimages::image8u overlay_simplified_segments(const libimages::image8u& input,
                                               const std::vector<std::pair<point2i, point2i>>& segments,
                                               const DrawParams& p);

// Draw red NN connections on top of an RGB image.
// - nn_B_to_A[i] = j means draw red line from seg[i].b to seg[j].a
// - nn_A_to_B[i] = j means draw red line from seg[i].a to seg[j].b
void draw_nn_connections_red_inplace(libimages::image8u& rgb_overlay,
                                     const std::vector<std::pair<point2i, point2i>>& segments,
                                     const std::vector<int>& nn_B_to_A,
                                     const std::vector<int>& nn_A_to_B,
                                     int thickness);

// Helpers: build 1-channel masks for starts/ends.
libimages::image8u mask_points(const int w, const int h, const std::vector<point2i>& pts);

} // namespace draw
