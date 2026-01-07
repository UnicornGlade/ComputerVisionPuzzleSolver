#include "find_mask_borders.h"

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_mask(int w, int h) {
    image8u m(w, h, 1);
    m.fill(static_cast<std::uint8_t>(0));
    return m;
}

static image8u make_rgb(int w, int h) {
    image8u img(w, h, 3);
    img.fill(static_cast<std::uint8_t>(0));
    return img;
}

static void draw_rect_mask(image8u& m, int x0, int y0, int x1, int y1, std::uint8_t v = 255) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i) m(j, i) = v;
}

static void draw_rect_rgb(image8u& img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i) {
            img(j, i, 0) = r;
            img(j, i, 1) = g;
            img(j, i, 2) = b;
        }
}

static int count_255(const image8u& m) {
    int cnt = 0;
    for (int j = 0; j < m.height(); ++j)
        for (int i = 0; i < m.width(); ++i)
            if (m(j, i) == 255) ++cnt;
    return cnt;
}

TEST(find_mask_borders, BorderMask_Rectangle) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case00_border";
    fs::create_directories(dir);

    image8u m = make_mask(80, 60);
    draw_rect_mask(m, 10, 8, 60, 45, 255);

    const auto b = find_mask_borders::border_pixels_mask(m, 2);

    libimages::debug_io::dump_image((dir / "00_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "01_border.png").string(), b, true, true);

    EXPECT_EQ(m(20, 20), 255);
    EXPECT_EQ(b(20, 20), 0);      // interior is not border
    EXPECT_EQ(b(8, 10), 255);     // top edge
    EXPECT_EQ(b(45, 60), 255);    // right edge
    EXPECT_EQ(b(0, 0), 0);        // background
    EXPECT_GT(count_255(b), 0);
}

TEST(find_mask_borders, SplitIntoSides_Rectangle) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case01_split";
    fs::create_directories(dir);

    image8u m = make_mask(120, 90);
    draw_rect_mask(m, 20, 15, 95, 70, 255);

    const auto b = find_mask_borders::border_pixels_mask(m, 2);
    const auto sides = find_mask_borders::split_border_into_sides(m, b, 4);

    libimages::debug_io::dump_image((dir / "00_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "01_border.png").string(), b, true, true);

    image8u dbg = make_rgb(m.width(), m.height());
    // visualize just sides on black
    dbg = find_mask_borders::visualize_sides_overlay(dbg, sides, 123, 1);
    libimages::debug_io::dump_image((dir / "02_sides_overlay.png").string(), dbg, true, true);

    EXPECT_GE(sides.size(), 1u);
    EXPECT_LE(sides.size(), 4u);

    int sum = 0;
    for (const auto& s : sides) sum += static_cast<int>(s.size());
    EXPECT_GT(sum, 0);
    EXPECT_GE(sum, static_cast<int>(0.7 * count_255(b))); // rough sanity
}

TEST(find_mask_borders, SampleColors_Debug) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case02_sample";
    fs::create_directories(dir);

    // Synthetic RGB image with a colored rectangle object.
    image8u img = make_rgb(140, 100);
    image8u m = make_mask(140, 100);
    draw_rect_mask(m, 25, 20, 110, 80, 255);

    // Paint object with varying color so sampling isn't trivial.
    draw_rect_rgb(img, 25, 20, 110, 80, 80, 80, 80);
    for (int i = 25; i <= 110; ++i) {
        img(20, i, 0) = 255; // top edge red
        img(80, i, 2) = 255; // bottom edge blue
    }
    for (int j = 20; j <= 80; ++j) {
        img(j, 25, 1) = 255; // left edge green
        img(j, 110, 0) = 255; img(j, 110, 1) = 255; // right edge yellow
    }

    const auto b = find_mask_borders::border_pixels_mask(m, 2);
    const auto sides = find_mask_borders::split_border_into_sides(m, b, 4);

    find_mask_borders::SamplingDebugParams dp;
    dp.out_dir = dir;
    dp.prefix = "obj0000_";
    dp.dump_ext = ".png";
    dp.force_overwrite = true;
    dp.verbose = false;
    dp.point_radius = 1;

    const auto samples = find_mask_borders::sample_sides_colors(img, sides, 10, &dp);
    const auto kxl = find_mask_borders::make_kxl_image(samples);

    libimages::debug_io::dump_image((dir / "00_image.png").string(), img, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "02_border.png").string(), b, true, true);
    libimages::debug_io::dump_image((dir / "03_sides_overlay.png").string(),
                                    find_mask_borders::visualize_sides_overlay(img, sides, 777, 2), true, true);
    libimages::debug_io::dump_image((dir / "04_samples_kxl.png").string(), kxl, true, true);

    ASSERT_EQ(samples.size(), sides.size());
    for (const auto& row : samples) EXPECT_EQ(row.size(), 10u);
}
