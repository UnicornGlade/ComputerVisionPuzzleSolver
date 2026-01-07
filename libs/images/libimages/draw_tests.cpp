#include "draw.h"

#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static int count_pure_red(const image8u& rgb) {
    int cnt = 0;
    for (int j = 0; j < rgb.height(); ++j)
        for (int i = 0; i < rgb.width(); ++i) {
            if (rgb(j, i, 0) == 255 && rgb(j, i, 1) == 0 && rgb(j, i, 2) == 0) ++cnt;
        }
    return cnt;
}

TEST(draw, OverlaySegmentsPixelsSmoke) {
    const fs::path dir = "debug-unit-tests/draw/case00";
    fs::create_directories(dir);

    image8u img(64, 48, 3);
    img.fill(static_cast<std::uint8_t>(200));

    std::vector<point2i> sp;
    for (int i = 10; i < 30; ++i) sp.push_back(point2i{i, 20});

    std::vector<std::vector<point2i>> segs = {sp};

    const auto out = draw::overlay_segments_pixels(img, segs, 123);
    libimages::debug_io::dump_image((dir / "00_overlay_segments.png").string(), out, true, true);

    // Expect some pixel to differ from darkened background.
    bool any_changed = false;
    for (int j = 0; j < out.height(); ++j) {
        for (int i = 0; i < out.width(); ++i) {
            if (out(j, i, 0) != 100 || out(j, i, 1) != 100 || out(j, i, 2) != 100) {
                any_changed = true;
                break;
            }
        }
        if (any_changed) break;
    }
    EXPECT_TRUE(any_changed);
}

TEST(draw, NNConnectionsIncreasePureRedCount) {
    const fs::path dir = "debug-unit-tests/draw/case01";
    fs::create_directories(dir);

    image8u img(80, 60, 3);
    img.fill(static_cast<std::uint8_t>(0));

    std::vector<std::pair<point2i, point2i>> segs;
    segs.push_back({point2i{10, 10}, point2i{30, 10}});
    segs.push_back({point2i{50, 12}, point2i{70, 12}});

    draw::DrawParams p;
    p.seed = 1;

    auto overlay = draw::overlay_simplified_segments(img, segs, p);
    const int red_before = count_pure_red(overlay);

    std::vector<int> nn_B_to_A = {1, 0};
    std::vector<int> nn_A_to_B = {1, 0};
    draw::draw_nn_connections_red_inplace(overlay, segs, nn_B_to_A, nn_A_to_B, 1);

    const int red_after = count_pure_red(overlay);
    libimages::debug_io::dump_image((dir / "00_overlay_with_nn.png").string(), overlay, true, true);

    EXPECT_GT(red_after, red_before);
}
