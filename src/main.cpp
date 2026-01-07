#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>
#include <libimages/algorithms/downscale.h>

#include "algorithms/find_segments.h"
#include "algorithms/find_lines.h"

namespace fs = std::filesystem;

namespace cfg {
inline constexpr float kDownscaleRatio = 8.0f;
}

using libimages::image8u;

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  app <input.(png|jpg|jpeg)> <output_dir> [debug_segments_dir]\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        const bool has_debug = (argc >= 4);
        const fs::path debug_dir = has_debug ? fs::path(argv[3]) : fs::path();

        image8u input = libimages::load_image(input_path.string());
        libimages::debug_io::dump_image((out_dir / "00_input.png").string(), input, true, true);

        input = libimages::downscale_area(input, cfg::kDownscaleRatio);
        libimages::debug_io::dump_image((out_dir / "01_input_downscaled.png").string(), input, true, true);

        find_segments::Params sp; // defaults
        find_segments::DebugParams dbg;
        find_segments::DebugParams* dbg_ptr = nullptr;
        if (has_debug) {
            dbg.out_dir = debug_dir;
            dbg.dump_ext = ".png";
            dbg.verbose = true;
            dbg.force_overwrite = true;
            dbg.large_component_min_size = 100;
            dbg_ptr = &dbg;
        }

        const auto segments = find_segments::find_segments(input, sp, dbg_ptr);

        const auto seg_overlay = find_segments::visualize_segments_overlay(input, segments, {});
        libimages::debug_io::dump_image((out_dir / "02_segments_overlay.png").string(), seg_overlay, true, true);

        std::cout << "segments: " << segments.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
