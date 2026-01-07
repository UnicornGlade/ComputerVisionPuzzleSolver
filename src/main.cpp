#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <libbase/runtime_assert.h>

#include <libimages/algorithms/downscale.h>
#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include "algorithms/background_masking.h"

namespace fs = std::filesystem;

namespace cfg {

// 1) Load + downscale
inline constexpr float kDownscaleRatio = 8.0f;

// 2) Blur
inline constexpr float kGaussianSigma = 1.2f; // bigger -> stronger smoothing (less noise, less details)

// 3) Mask refinement (can be set to 0 to disable)
inline constexpr int kMaskDilateStrength = 2; // bigger -> fills gaps more
inline constexpr int kMaskErodeStrength = 2;  // bigger -> removes thin parts more (after dilation: closing)

// High-level outputs
inline constexpr const char* kOut00 = "00_input_downscaled.png";
inline constexpr const char* kOut01 = "01_blur_gray.png";
inline constexpr const char* kOut02 = "02_foreground_mask.png";

// Low-level folders
inline constexpr const char* kDbg01 = "01_load_downscale";
inline constexpr const char* kDbg02 = "02_blur";
inline constexpr const char* kDbg03 = "03_background_mask";

// Low-level file names (inside stage folders)
inline constexpr const char* kLL00 = "00_input_downscaled.png";
inline constexpr const char* kLL10 = "10_gray_u8.png";
inline constexpr const char* kLL20 = "20_blur_gray_u8.png";
inline constexpr const char* kLL30 = "30_mask_raw.png";
inline constexpr const char* kLL31 = "31_mask_refined.png";

} // namespace cfg

using libimages::image8u;
using libimages::image32f;

static image8u gray32f_to_u8_clamp(const image32f& gray) {
    rassert(gray.channels() == 1, "gray32f_to_u8_clamp expects 1-channel image32f", gray.channels());

    image8u out(gray.width(), gray.height(), 1);
    for (int j = 0; j < gray.height(); ++j) {
        for (int i = 0; i < gray.width(); ++i) {
            float v = gray(j, i);
            if (std::isnan(v) || std::isinf(v)) v = 0.0f;
            v = std::clamp(v, 0.0f, 255.0f);
            out(j, i) = static_cast<std::uint8_t>(std::lround(v));
        }
    }
    return out;
}

static image8u make_raw_threshold_mask(const image8u& gray_u8, float thr) {
    rassert(gray_u8.channels() == 1, "make_raw_threshold_mask expects 1-channel image8u", gray_u8.channels());
    image8u m(gray_u8.width(), gray_u8.height(), 1);
    for (int j = 0; j < gray_u8.height(); ++j) {
        for (int i = 0; i < gray_u8.width(); ++i) {
            m(j, i) = (static_cast<float>(gray_u8(j, i)) > thr) ? 255 : 0;
        }
    }
    return m;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  app <input.(png|jpg|jpeg)> <output_dir> [debug_root]\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        const fs::path debug_root = (argc >= 4) ? fs::path(argv[3]) : (out_dir / "debug-low");
        fs::create_directories(debug_root);

        const fs::path dbg1 = debug_root / cfg::kDbg01;
        const fs::path dbg2 = debug_root / cfg::kDbg02;
        const fs::path dbg3 = debug_root / cfg::kDbg03;
        fs::create_directories(dbg1);
        fs::create_directories(dbg2);
        fs::create_directories(dbg3);

        // =========================================================
        // 1) Read input image and downscale
        // =========================================================
        image8u input = libimages::load_image(input_path.string());
        input = libimages::downscale_area(input, cfg::kDownscaleRatio);

        libimages::debug_io::dump_image((out_dir / cfg::kOut00).string(), input, true, true);
        libimages::debug_io::dump_image((dbg1 / cfg::kLL00).string(), input, true, true);

        // =========================================================
        // 2) Blur (on grayscale)
        // =========================================================
        const image32f gray_f = libimages::to_grayscale_float(input);
        const image8u gray_u8 = gray32f_to_u8_clamp(gray_f);
        libimages::debug_io::dump_image((dbg2 / cfg::kLL10).string(), gray_u8, true, true);

        const image32f blur_f = libimages::gaussian_blur_gray(gray_f, cfg::kGaussianSigma);
        const image8u blur_u8 = gray32f_to_u8_clamp(blur_f);

        libimages::debug_io::dump_image((out_dir / cfg::kOut01).string(), blur_u8, true, true);
        libimages::debug_io::dump_image((dbg2 / cfg::kLL20).string(), blur_u8, true, true);

        // =========================================================
        // 3) Build foreground mask using background masking
        // =========================================================
        background_masking::Params mp;
        mp.dilate_strength = cfg::kMaskDilateStrength;
        mp.erode_strength = cfg::kMaskErodeStrength;

        // Low-level: show threshold + raw mask (before morphology)
        const float thr = background_masking::estimate_background_threshold(blur_u8);
        const image8u mask_raw = make_raw_threshold_mask(blur_u8, thr);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL30).string(), mask_raw, true, true);

        // Final (with morphology refinement)
        const image8u mask = background_masking::build_foreground_mask(blur_u8, mp);

        libimages::debug_io::dump_image((out_dir / cfg::kOut02).string(), mask, true, true);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL31).string(), mask, true, true);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
