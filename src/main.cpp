#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include "algorithms/find_segments.h"
#include "libimages/algorithms/downscale.h"

namespace fs = std::filesystem;

namespace cfg {
inline constexpr float kDownscaleRatio = 8.0f; // 1 -> no scale, >1 treated as 1/kDownscaleRatio
}

using libimages::image8u;

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  segments_app <input.(png|jpg|jpeg)> <output_dir> [debug_dir]\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        const bool has_debug = (argc >= 4);
        const fs::path debug_dir = has_debug ? fs::path(argv[3]) : fs::path();

        image8u input = libimages::load_image(input_path.string());
        libimages::debug_io::dump_image((out_dir / "00_input.png").string(), input, /*verbose=*/true, /*force=*/true);

        // Downscale in main (so coordinates in segments match the image passed into find_segments()).
        input = libimages::downscale_area(input, cfg::kDownscaleRatio);
        libimages::debug_io::dump_image((out_dir / "01_input_downscaled.png").string(), input, /*verbose=*/true,
                                        /*force=*/true);

        find_segments::Params params; // defaults are "current"
        // Example tweak (optional):
        // params.min_segment_pixels = 25;

        find_segments::DebugParams dbg;
        find_segments::DebugParams *dbg_ptr = nullptr;
        if (has_debug) {
            dbg.out_dir = debug_dir;
            dbg.dump_ext = ".png";
            dbg.verbose = true;
            dbg.force_overwrite = true;
            dbg.large_component_min_size = 100;
            dbg_ptr = &dbg;
        }

        const std::vector<find_segments::SegmentPixels> segs = find_segments::find_segments(input, params, dbg_ptr);

        find_segments::VisualizeParams vis;
        vis.min_pixels = 2;
        vis.min_median_magnitude = 0.0f;
        vis.max_median_angle_deviation_deg = 180.0f;
        vis.seed = 123;

        const image8u overlay = find_segments::visualize_segments_overlay(input, segs, vis);
        libimages::debug_io::dump_image((out_dir / "02_segments_overlay.png").string(), overlay, /*verbose=*/true,
                                        /*force=*/true);

        // Large-only overlay for convenience.
        find_segments::VisualizeParams vis_large = vis;
        vis_large.min_pixels = 100;
        const image8u overlay_large = find_segments::visualize_segments_overlay(input, segs, vis_large);
        libimages::debug_io::dump_image((out_dir / "03_segments_overlay_large.png").string(), overlay_large,
                                        /*verbose=*/true, /*force=*/true);

        std::cout << "Segments: " << segs.size() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
