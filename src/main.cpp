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
#include "algorithms/find_lines.h"

namespace fs = std::filesystem;

namespace cfg {
inline constexpr float kDownscaleRatio = 8.0f;
}

using libimages::image8u;

static float normalize_downscale_ratio(float r) {
    rassert(r > 0.0f, "kDownscaleRatio must be > 0", r);
    if (r > 1.0f) r = 1.0f / r;
    rassert(r > 0.0f && r <= 1.0f, "normalized ratio must be in (0,1]", r);
    return r;
}

static image8u downscale_nearest(const image8u& img, float downscale_ratio) {
    const float s = normalize_downscale_ratio(downscale_ratio);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

    if (std::fabs(s - 1.0f) < 1e-7f) return img;

    const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
    const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

    image8u out(nw, nh, c);

    const float inv_s = 1.0f / s;
    for (int j = 0; j < nh; ++j) {
        const int sj = std::clamp(static_cast<int>(std::floor((static_cast<float>(j) + 0.5f) * inv_s)), 0, h - 1);
        for (int i = 0; i < nw; ++i) {
            const int si = std::clamp(static_cast<int>(std::floor((static_cast<float>(i) + 0.5f) * inv_s)), 0, w - 1);
            if (c == 1) {
                out(j, i) = img(sj, si);
            } else {
                for (int k = 0; k < c; ++k) out(j, i, k) = img(sj, si, k);
            }
        }
    }
    return out;
}

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

        input = downscale_nearest(input, cfg::kDownscaleRatio);
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

        // ---- NEW: find_lines stage ----
        find_lines::Params lp; // defaults
        const auto lines = find_lines::find_lines(segments, lp);

        const auto lines_overlay = find_lines::visualize_lines_overlay(input, lines, {});
        libimages::debug_io::dump_image((out_dir / "03_lines_overlay.png").string(), lines_overlay, true, true);

        std::cout << "segments: " << segments.size() << "\n";
        std::cout << "lines:    " << lines.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
