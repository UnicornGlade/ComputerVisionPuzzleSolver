#pragma once

#include <cstdint>
#include <vector>

#include <libimages/image.h>

#include "find_segments.h"

namespace find_lines {

struct LinePixels {
    // Union of pixels from all inlier segments.
    std::vector<point2i> pixels;

    // Line model used for this output line.
    // Point is the medoid (pixel coordinates) of the best candidate segment in that iteration.
    point2i medoid{0, 0};

    // Unit direction of the line (dx, dy).
    float dir_x = 1.0f;
    float dir_y = 0.0f;

    int total_votes_pixels = 0; // sum of pixel-count votes from inlier segments
    int inlier_segments = 0;    // how many segments contributed
};

struct Params {
    // Candidate selection.
    int min_candidate_pixels = 10; // Larger -> fewer candidates. Smaller -> more candidates (and more false positives).

    // Voting conditions.
    float max_angle_diff_deg = 5.0f;  // Larger -> merge less-parallel segments.
    float max_dist_px = 5.0f;         // Larger -> allow farther segments to vote as inliers.
    int min_total_votes_pixels = 50;  // Larger -> require stronger support to accept a line.

    // RANSAC-like outer loop limit.
    int max_iterations = 1000;
};

struct VisualizeParams {
    int min_pixels = 2;
    std::uint32_t seed = 123;
};

// Input: segments from find_segments.
// Output: bigger line-like segments by iterative best-consensus removal.
std::vector<LinePixels> find_lines(const std::vector<find_segments::SegmentPixels>& segments,
                                  const Params& params = Params{});

// Debug visualization helper: darken input image by 2x and paint line pixels with random colors.
libimages::image8u visualize_lines_overlay(const libimages::image8u& input,
                                          const std::vector<LinePixels>& lines,
                                          const VisualizeParams& vis = VisualizeParams{});

} // namespace find_lines
