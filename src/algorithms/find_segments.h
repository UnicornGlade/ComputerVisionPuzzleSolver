#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <libbase/point2.h>
#include <libimages/image.h>

namespace find_segments {

struct SegmentPixels {
    std::vector<point2i> pixels;

    float median_magnitude = 0.0f;
    float median_abs_dev_magnitude = 0.0f;

    float median_angle_deg = 0.0f;         // in [0, 360)
    float median_abs_dev_angle_deg = 0.0f; // in degrees, [0, 180]
};

struct Params {
    // Preprocessing (inside find_segments).
    float gaussian_sigma = 0.01f; // Larger -> smoother, fewer/noisier edges removed. Smaller -> sharper/noisier.

    // Olson 2011 Eq(1) clustering thresholds.
    float cluster_KD = 1000.0f;  // Larger -> allow bigger angle spread -> more merges.
    float cluster_KM = 20000.0f; // Larger -> allow bigger magnitude spread -> more merges.

    // Output filtering: only components with at least this many pixels become "segments".
    int min_segment_pixels = 25; // Larger -> fewer segments, smaller -> more (incl. junk).
};

struct DebugParams {
    // If nullptr -> no debug output at all.
    std::filesystem::path out_dir;

    std::string dump_ext = ".png";
    bool verbose = true;
    bool force_overwrite = true;

    // For "large components" debug visualization.
    int large_component_min_size = 100;
};

struct VisualizeParams {
    int min_pixels = 2;
    float min_median_magnitude = 0.0f;
    float max_median_angle_deviation_deg = 180.0f;

    std::uint32_t seed = 0;
};

// Finds segments (pixel sets) on the given image (already downscaled if needed).
std::vector<SegmentPixels> find_segments(const libimages::image8u &input, const Params &params = Params{},
                                         const DebugParams *debug = nullptr);

// Darken input (x0.5) and paint segments with random colors (filtered by params).
libimages::image8u visualize_segments_overlay(const libimages::image8u &input,
                                              const std::vector<SegmentPixels> &segments,
                                              const VisualizeParams &vis = VisualizeParams{});

} // namespace find_segments
