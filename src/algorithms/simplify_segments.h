#pragma once

#include <cstdint>
#include <vector>

#include <libimages/image.h>

#include "find_segments.h"

namespace simplify_segments {

struct Segment {
    point2i a;
    point2i b;
};

struct VisualizeParams {
    std::uint32_t seed = 123;
    int thickness = 1; // 1 -> single-pixel line, >1 -> thicker (square brush)
};

// Simplify one pixel-set segment into a single line segment [A,B].
Segment simplify_segment(const find_segments::SegmentPixels& s);

// Convenience: simplify all.
std::vector<Segment> simplify_segments(const std::vector<find_segments::SegmentPixels>& segments);

// Visualization: darken input by 2x and draw segments (A->B) in random colors.
libimages::image8u visualize_segments_overlay(const libimages::image8u& input,
                                             const std::vector<Segment>& segments,
                                             const VisualizeParams& vis = VisualizeParams{});

} // namespace simplify_segments
