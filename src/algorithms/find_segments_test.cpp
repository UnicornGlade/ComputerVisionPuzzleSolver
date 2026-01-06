#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

#include "find_segments.h"

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_vertical_step_edge(int w, int h) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::uint8_t v = (i < w / 2) ? 0 : 255;
            img(j, i, 0) = v;
            img(j, i, 1) = v;
            img(j, i, 2) = v;
        }
    }
    return img;
}

static image8u make_rectangle_border(int w, int h, int x0, int y0, int x1, int y1) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            img(j, i, 0) = img(j, i, 1) = img(j, i, 2) = 0;

    for (int i = x0; i <= x1; ++i) {
        img(y0, i, 0) = img(y0, i, 1) = img(y0, i, 2) = 255;
        img(y1, i, 0) = img(y1, i, 1) = img(y1, i, 2) = 255;
    }
    for (int j = y0; j <= y1; ++j) {
        img(j, x0, 0) = img(j, x0, 1) = img(j, x0, 2) = 255;
        img(j, x1, 0) = img(j, x1, 1) = img(j, x1, 2) = 255;
    }
    return img;
}

static void run_case(const fs::path& root, const std::string& name, const image8u& img) {
    const fs::path dir = root / name;
    fs::create_directories(dir);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, /*verbose=*/true, /*force=*/true);

    find_segments::Params p;
    // Allow smaller segments for synthetic patterns.
    p.min_segment_pixels = 10;

    const auto segs = find_segments::find_segments(img, p, /*debug=*/nullptr);
    ASSERT_FALSE(segs.empty()) << "Expected at least one segment in " << name;

    const auto overlay = find_segments::visualize_segments_overlay(img, segs, {});
    libimages::debug_io::dump_image((dir / "01_segments_overlay.png").string(), overlay, /*verbose=*/true, /*force=*/true);

    find_segments::VisualizeParams vp_large;
    vp_large.min_pixels = 50;
    const auto overlay_large = find_segments::visualize_segments_overlay(img, segs, vp_large);
    libimages::debug_io::dump_image((dir / "02_segments_overlay_large.png").string(), overlay_large, /*verbose=*/true,
                                    /*force=*/true);
}

TEST(find_segments, VerticalStepEdge) {
    const fs::path root = "debug-unit-tests/find-segments";
    fs::create_directories(root);
    run_case(root, "case00_vertical_step_edge", make_vertical_step_edge(160, 120));
}

TEST(find_segments, RectangleBorder) {
    const fs::path root = "debug-unit-tests/find-segments";
    fs::create_directories(root);
    run_case(root, "case01_rectangle_border", make_rectangle_border(200, 150, 30, 25, 170, 120));
}
