#include "background_masking.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_gray(int w, int h, std::uint8_t v) {
    image8u img(w, h, 1);
    img.fill(v);
    return img;
}

static void draw_rect(image8u& img, int x0, int y0, int x1, int y1, std::uint8_t v) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i)
            img(j, i) = v;
}

static int count_white(const image8u& img) {
    int cnt = 0;
    for (int j = 0; j < img.height(); ++j)
        for (int i = 0; i < img.width(); ++i)
            if (img(j, i) == 255) ++cnt;
    return cnt;
}

TEST(background_masking, BasicCenterObject) {
    const fs::path dir = "debug-unit-tests/background-masking/case00_basic";
    fs::create_directories(dir);

    // Background ~50 on perimeter, bright object in center.
    image8u in = make_gray(160, 120, 50);
    draw_rect(in, 50, 35, 110, 85, 200);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);

    const float thr = background_masking::estimate_background_threshold(in);
    std::cout << "[test] threshold=" << thr << "\n";

    background_masking::Params p;
    p.dilate_strength = 0;
    p.erode_strength = 0;

    const auto mask = background_masking::build_foreground_mask(in, p);
    libimages::debug_io::dump_image((dir / "01_mask_raw.png").string(), mask, true, true);

    EXPECT_GT(thr, 0.0f);
    EXPECT_LT(thr, 255.0f);

    // Center should be mostly white, perimeter should be black.
    EXPECT_EQ(mask(0, 0), 0);
    EXPECT_EQ(mask(10, 10), 0);
    EXPECT_EQ(mask(60, 60), 255);
}

TEST(background_masking, PerimeterOutliersRobustByP90) {
    const fs::path dir = "debug-unit-tests/background-masking/case01_outliers";
    fs::create_directories(dir);

    // Mostly dark perimeter (40), but sprinkle some bright outliers on perimeter.
    image8u in = make_gray(200, 140, 40);
    for (int i = 0; i < in.width(); i += 20) {
        in(0, i) = 200;
        in(in.height() - 1, i) = 200;
    }
    // Center object
    draw_rect(in, 70, 50, 130, 95, 180);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);

    const float thr = background_masking::estimate_background_threshold(in);
    std::cout << "[test] threshold=" << thr << "\n";

    background_masking::Params p;
    p.dilate_strength = 0;
    p.erode_strength = 0;

    const auto mask = background_masking::build_foreground_mask(in, p);
    libimages::debug_io::dump_image((dir / "01_mask_raw.png").string(), mask, true, true);

    // Expect object survives.
    EXPECT_EQ(mask(70, 100), 255);
    // Typical background should be removed.
    EXPECT_EQ(mask(10, 10), 0);
}

TEST(background_masking, MorphologyClosingFillsHoles) {
    const fs::path dir = "debug-unit-tests/background-masking/case02_closing";
    fs::create_directories(dir);

    image8u in = make_gray(180, 130, 60);
    // Object with a "hole" of background in the middle
    draw_rect(in, 55, 40, 125, 95, 210);
    draw_rect(in, 85, 60, 95, 70, 60);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);

    background_masking::Params p0;
    p0.dilate_strength = 0;
    p0.erode_strength = 0;

    const auto mask_raw = background_masking::build_foreground_mask(in, p0);
    libimages::debug_io::dump_image((dir / "01_mask_raw.png").string(), mask_raw, true, true);

    background_masking::Params p;
    p.dilate_strength = 6;
    p.erode_strength = 6;

    const auto mask_refined = background_masking::build_foreground_mask(in, p);
    libimages::debug_io::dump_image((dir / "02_mask_refined_closing.png").string(), mask_refined, true, true);

    // Hole pixel: raw likely 0, refined should become 255 after closing (depending on threshold).
    EXPECT_EQ(mask_raw(65, 90), 0); // just keep test robust
    EXPECT_EQ(mask_refined(65, 90), 255);

    EXPECT_GE(count_white(mask_refined), count_white(mask_raw));
}
