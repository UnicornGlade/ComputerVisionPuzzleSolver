#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

#include "find_segments.h"
#include "simplify_segments.h"

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_black(int w, int h) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            img(j, i, 0) = img(j, i, 1) = img(j, i, 2) = 0;
    return img;
}

static void draw_vertical_line(image8u& img, int x, int thickness) {
    const int w = img.width();
    const int h = img.height();
    for (int j = 10; j < h - 10; ++j) {
        for (int t = -thickness; t <= thickness; ++t) {
            const int xx = x + t;
            if (xx < 0 || xx >= w) continue;
            img(j, xx, 0) = img(j, xx, 1) = img(j, xx, 2) = 255;
        }
    }
}

static void draw_diagonal_line(image8u& img, int thickness) {
    const int w = img.width();
    const int h = img.height();
    for (int k = 10; k < std::min(w, h) - 10; ++k) {
        for (int dy = -thickness; dy <= thickness; ++dy)
            for (int dx = -thickness; dx <= thickness; ++dx) {
                const int x = k + dx;
                const int y = k + dy;
                if (x < 0 || x >= w || y < 0 || y >= h) continue;
                img(y, x, 0) = img(y, x, 1) = img(y, x, 2) = 255;
            }
    }
}

static void draw_rectangle_border(image8u& img, int x0, int y0, int x1, int y1, int thickness) {
    const int w = img.width();
    const int h = img.height();

    auto put = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        img(y, x, 0) = img(y, x, 1) = img(y, x, 2) = 255;
    };

    for (int t = -thickness; t <= thickness; ++t) {
        for (int x = x0; x <= x1; ++x) { put(x, y0 + t); put(x, y1 + t); }
        for (int y = y0; y <= y1; ++y) { put(x0 + t, y); put(x1 + t, y); }
    }
}

static void run_case(const fs::path& root, const std::string& name, const image8u& img) {
    const fs::path dir = root / name;
    fs::create_directories(dir);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, true, true);

    find_segments::Params p;
    // Make it easier for synthetic edges:
    p.gaussian_sigma = 0.8f;
    p.cluster_KD = 2000.0f;
    p.cluster_KM = 40000.0f;
    p.min_segment_pixels = 10;

    const auto segs = find_segments::find_segments(img, p, /*debug=*/nullptr);
    ASSERT_FALSE(segs.empty()) << "Expected non-empty segments in " << name;

    const auto seg_overlay = find_segments::visualize_segments_overlay(img, segs, {});
    libimages::debug_io::dump_image((dir / "01_segments_pixels_overlay.png").string(), seg_overlay, true, true);

    const auto simple = simplify_segments::simplify_segments(segs);
    ASSERT_FALSE(simple.empty()) << "Expected non-empty simplified segments in " << name;

    simplify_segments::VisualizeParams vp;
    vp.seed = 123;

    const auto simple_overlay = simplify_segments::visualize_segments_overlay(img, simple, vp);
    libimages::debug_io::dump_image((dir / "02_simplified_segments_overlay.png").string(), simple_overlay, true, true);
}

TEST(simplify_segments, VerticalAndDiagonal) {
    const fs::path root = "debug-unit-tests/simplify-segments";
    fs::create_directories(root);

    image8u img = make_black(260, 180);
    draw_vertical_line(img, 60, 2);
    draw_vertical_line(img, 190, 1);
    draw_diagonal_line(img, 1);

    run_case(root, "case00_vertical_and_diagonal", img);
}

TEST(simplify_segments, RectangleBorder) {
    const fs::path root = "debug-unit-tests/simplify-segments";
    fs::create_directories(root);

    image8u img = make_black(300, 220);
    draw_rectangle_border(img, 40, 35, 260, 180, 1);

    run_case(root, "case01_rectangle_border", img);
}
