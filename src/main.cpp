#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

#include <libimages/algorithms/downscale.h>
#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include "algorithms/background_masking.h"
#include "algorithms/utils.h"
#include "algorithms/find_mask_borders.h"

namespace fs = std::filesystem;

namespace cfg {

// 1) Load + downscale
inline constexpr float kDownscaleRatio = 8.0f;

// 2) Blur
inline constexpr float kGaussianSigma = 1.2f;

// 3) Mask refinement
inline constexpr int kMaskDilateStrength = 2;
inline constexpr int kMaskErodeStrength = 2;

// 4) Extract objects
inline constexpr bool kObjectsEightConnected = true;

// 5) Border/sides/sampling
inline constexpr int kBorderMinBgNeighbors = 2; // >=2 background neighbors => border pixel
inline constexpr int kBorderSideCount = 4;      // K
inline constexpr int kSamplesPerSide = 10;      // L

// High-level outputs
inline constexpr const char* kOut00 = "00_input_downscaled.png";
inline constexpr const char* kOut01 = "01_blur_gray.png";
inline constexpr const char* kOut02 = "02_foreground_mask.png";
inline constexpr const char* kOut03 = "03_objects_overlay.png";

// Low-level folders
inline constexpr const char* kDbg01 = "01_load_downscale";
inline constexpr const char* kDbg02 = "02_blur";
inline constexpr const char* kDbg03 = "03_background_mask";
inline constexpr const char* kDbg04 = "04_extract_objects";
inline constexpr const char* kDbg05 = "05_find_mask_borders";

// Low-level file names
inline constexpr const char* kLL00 = "00_input_downscaled.png";
inline constexpr const char* kLL10 = "10_gray_u8.png";
inline constexpr const char* kLL20 = "20_blur_gray_u8.png";
inline constexpr const char* kLL30 = "30_mask_raw.png";
inline constexpr const char* kLL31 = "31_mask_refined.png";
inline constexpr const char* kLL40 = "40_objects_overlay.png";

inline constexpr std::uint32_t kVizSeed = 123;

} // namespace cfg

using libimages::image8u;
using libimages::image32f;

static std::string idx4(std::size_t k) {
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << k;
    return ss.str();
}

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

static image8u make_dark_rgb(const image8u& input) {
    const int w = input.width();
    const int h = input.height();
    rassert(input.channels() == 1 || input.channels() == 3 || input.channels() == 4, "Unsupported input channels",
            input.channels());

    image8u out(w, h, 3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            std::uint8_t r = 0, g = 0, b = 0;
            if (input.channels() == 1) {
                const std::uint8_t v = input(j, i);
                r = g = b = static_cast<std::uint8_t>(v / 2);
            } else {
                r = static_cast<std::uint8_t>(input(j, i, 0) / 2);
                g = static_cast<std::uint8_t>(input(j, i, 1) / 2);
                b = static_cast<std::uint8_t>(input(j, i, 2) / 2);
            }
            out(j, i, 0) = r;
            out(j, i, 1) = g;
            out(j, i, 2) = b;
        }
    }
    return out;
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
        const fs::path dbg4 = debug_root / cfg::kDbg04;
        const fs::path dbg5 = debug_root / cfg::kDbg05;
        fs::create_directories(dbg1);
        fs::create_directories(dbg2);
        fs::create_directories(dbg3);
        fs::create_directories(dbg4);
        fs::create_directories(dbg5);

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

        const float thr = background_masking::estimate_background_threshold(blur_u8);
        const image8u mask_raw = make_raw_threshold_mask(blur_u8, thr);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL30).string(), mask_raw, true, true);

        const image8u mask = background_masking::build_foreground_mask(blur_u8, mp);

        libimages::debug_io::dump_image((out_dir / cfg::kOut02).string(), mask, true, true);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL31).string(), mask, true, true);

        // =========================================================
        // 4) Split into connected foreground objects + save each as image+mask (flat)
        // =========================================================
        utils::ExtractParams ep;
        ep.eight_connected = cfg::kObjectsEightConnected;

        const auto objects = utils::extract_objects_by_mask<std::uint8_t>(input, mask, ep);

        const fs::path out_objs = out_dir / "objects";
        const fs::path dbg_objs = dbg4 / "objects";
        fs::create_directories(out_objs);
        fs::create_directories(dbg_objs);

        image8u hi_overlay = make_dark_rgb(input);
        FastRandom rng(cfg::kVizSeed);

        for (std::size_t k = 0; k < objects.size(); ++k) {
            const auto& obj = objects[k];
            const std::string pre = idx4(k);

            libimages::debug_io::dump_image((out_objs / (pre + "image.png")).string(), obj.image, true, true);
            libimages::debug_io::dump_image((out_objs / (pre + "mask.png")).string(), obj.mask, true, true);

            libimages::debug_io::dump_image((dbg_objs / (pre + "image.png")).string(), obj.image, true, true);
            libimages::debug_io::dump_image((dbg_objs / (pre + "mask.png")).string(), obj.mask, true, true);

            const std::uint32_t rv = rng.nextU32();
            const std::uint8_t cr = static_cast<std::uint8_t>(rv & 0xFFu);
            const std::uint8_t cg = static_cast<std::uint8_t>((rv >> 8) & 0xFFu);
            const std::uint8_t cb = static_cast<std::uint8_t>((rv >> 16) & 0xFFu);

            for (int jj = 0; jj < obj.mask.height(); ++jj) {
                for (int ii = 0; ii < obj.mask.width(); ++ii) {
                    if (obj.mask(jj, ii) != 255) continue;
                    const int y = obj.offset.y + jj;
                    const int x = obj.offset.x + ii;
                    if (x < 0 || x >= hi_overlay.width() || y < 0 || y >= hi_overlay.height()) continue;
                    hi_overlay(y, x, 0) = cr;
                    hi_overlay(y, x, 1) = cg;
                    hi_overlay(y, x, 2) = cb;
                }
            }
        }

        libimages::debug_io::dump_image((out_dir / cfg::kOut03).string(), hi_overlay, true, true);
        libimages::debug_io::dump_image((dbg4 / cfg::kLL40).string(), hi_overlay, true, true);

        // =========================================================
        // 5) NEW: For each object -> border mask -> K sides -> sample KxL colors
        // =========================================================
        const fs::path out_sides = out_dir / "objects_sides";
        const fs::path dbg_sides = dbg5 / "objects_sides";
        fs::create_directories(out_sides);
        fs::create_directories(dbg_sides);

        for (std::size_t k = 0; k < objects.size(); ++k) {
            const auto& obj = objects[k];
            const std::string pre = idx4(k);

            // 5.1 border pixels
            const image8u border = find_mask_borders::border_pixels_mask(obj.mask, cfg::kBorderMinBgNeighbors);
            libimages::debug_io::dump_image((out_sides / (pre + "border.png")).string(), border, true, true);
            libimages::debug_io::dump_image((dbg_sides / (pre + "border.png")).string(), border, true, true);

            // 5.2 split border into K sides
            const auto sides = find_mask_borders::split_border_into_sides(obj.mask, border, cfg::kBorderSideCount);

            const image8u sides_overlay =
                find_mask_borders::visualize_sides_overlay(obj.image, sides, cfg::kVizSeed ^ static_cast<std::uint32_t>(k), 2);

            libimages::debug_io::dump_image((out_sides / (pre + "sides_overlay.png")).string(), sides_overlay, true, true);
            libimages::debug_io::dump_image((dbg_sides / (pre + "sides_overlay.png")).string(), sides_overlay, true, true);

            // 5.3 sample colors along each side and dump debug with red sample points + blue donors
            find_mask_borders::SamplingDebugParams dp;
            dp.out_dir = dbg_sides;
            dp.prefix = pre + "_";
            dp.dump_ext = ".png";
            dp.force_overwrite = true;
            dp.verbose = false;
            dp.point_radius = 1;

            const auto samples = find_mask_borders::sample_sides_colors(obj.image, sides, cfg::kSamplesPerSide, &dp);
            const image8u kxl = find_mask_borders::make_kxl_image(samples);

            libimages::debug_io::dump_image((out_sides / (pre + "samples_kxl.png")).string(), kxl, true, true);
            libimages::debug_io::dump_image((dbg_sides / (pre + "samples_kxl.png")).string(), kxl, true, true);

            std::cout << "[obj " << pre << "] sides=" << sides.size()
                      << " K=" << cfg::kBorderSideCount << " L=" << cfg::kSamplesPerSide << "\n";
        }

        std::cout << "objects: " << objects.size() << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
