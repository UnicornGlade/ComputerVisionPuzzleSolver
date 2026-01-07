#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libimages/algorithms/sobel_gradients.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>

#include "non_maximum_suppression.h"

namespace fs = std::filesystem;

using libimages::Gradients;
using libimages::image32f;
using libimages::image8u;

static Gradients make_empty_gradients(int w, int h) {
    Gradients g{
        image32f(w, h, 1), // dx
        image32f(w, h, 1), // dy
        image32f(w, h, 1), // mag
        image32f(w, h, 1)  // angle (may be unused here)
    };
    g.dx.fill(0.0f);
    g.dy.fill(0.0f);
    g.mag.fill(0.0f);
    g.angle.fill(0.0f);
    return g;
}

static image32f mask_to_u8_vis(const image8u& m) {
    image32f out(m.width(), m.height(), 1);
    for (int j = 0; j < m.height(); ++j)
        for (int i = 0; i < m.width(); ++i)
            out(j, i) = (m(j, i) ? 255.0f : 0.0f);
    return out;
}

TEST(non_maximum_suppression, HorizontalDirectionKeepsStrictMax) {
    const fs::path dir = "debug-unit-tests/non_maximum_suppression/case00_horizontal";
    fs::create_directories(dir);

    auto g = make_empty_gradients(7, 7);

    // Direction: +x (right). So compare with left/right neighbors.
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i) {
            g.dx(j, i) = 1.0f;
            g.dy(j, i) = 0.0f;
        }

    // Create 1D ridge along row 3: [.. 5, 10, 5 ..]
    g.mag(3, 2) = 5.0f;
    g.mag(3, 3) = 10.0f; // should survive
    g.mag(3, 4) = 5.0f;

    const image8u m = libimages::non_maximum_suppression(g);
    EXPECT_EQ(m(3, 3), 255);
    EXPECT_EQ(m(3, 2), 0);
    EXPECT_EQ(m(3, 4), 0);

    libimages::debug_io::dump_image((dir / "00_mag.png").string(), g.mag, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask_to_u8_vis(m), true, true);
}

TEST(non_maximum_suppression, EqualNeighborNotSuppresses) {
    const fs::path dir = "debug-unit-tests/non_maximum_suppression/case01_equal";
    fs::create_directories(dir);

    auto g = make_empty_gradients(7, 7);
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i) {
            g.dx(j, i) = 1.0f;
            g.dy(j, i) = 0.0f;
        }

    // Center equals right neighbor -> must be suppressed (strict >).
    g.mag(3, 2) = 5.0f;
    g.mag(3, 3) = 10.0f;
    g.mag(3, 4) = 10.0f;

    const image8u m = libimages::non_maximum_suppression(g);
    EXPECT_EQ(m(3, 1), 0);
    EXPECT_EQ(m(3, 2), 0);
    EXPECT_EQ(m(3, 3), 255);
    EXPECT_EQ(m(3, 4), 255);
    EXPECT_EQ(m(3, 5), 0);

    libimages::debug_io::dump_image((dir / "00_mag.png").string(), g.mag, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask_to_u8_vis(m), true, true);
}

TEST(non_maximum_suppression, Diagonal45DirectionUsesDiagonalNeighbors) {
    const fs::path dir = "debug-unit-tests/non_maximum_suppression/case02_diag45";
    fs::create_directories(dir);

    auto g = make_empty_gradients(7, 7);

    // Direction: down-right (dx=1, dy=1) -> compare (j+1,i+1) and (j-1,i-1)
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i) {
            g.dx(j, i) = 1.0f;
            g.dy(j, i) = 1.0f;
        }

    g.mag(2, 2) = 5.0f;
    g.mag(3, 3) = 10.0f; // should survive
    g.mag(4, 4) = 6.0f;

    const image8u m = libimages::non_maximum_suppression(g);
    EXPECT_EQ(m(3, 3), 255);

    libimages::debug_io::dump_image((dir / "00_mag.png").string(), g.mag, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask_to_u8_vis(m), true, true);
}

TEST(non_maximum_suppression, ApplyMaskZeroesGradients) {
    const fs::path dir = "debug-unit-tests/non_maximum_suppression/case03_apply_mask";
    fs::create_directories(dir);

    auto g = make_empty_gradients(5, 5);
    for (int j = 0; j < 5; ++j)
        for (int i = 0; i < 5; ++i) {
            g.dx(j, i) = 1.0f;
            g.dy(j, i) = 2.0f;
            g.mag(j, i) = 3.0f;
            g.angle(j, i) = 4.0f;
        }

    image8u mask(5, 5, 1);
    mask.fill(static_cast<std::uint8_t>(255));
    mask(2, 2) = 0;

    libimages::apply_mask_inplace(g, mask);
    EXPECT_FLOAT_EQ(g.mag(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(g.dx(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(g.dy(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(g.angle(2, 2), 0.0f);

    libimages::debug_io::dump_image((dir / "00_mask.png").string(), mask_to_u8_vis(mask), true, true);
    libimages::debug_io::dump_image((dir / "01_mag_after.png").string(), g.mag, true, true);
}

TEST(non_maximum_suppression, MagnitudeThresholdMask) {
    const fs::path dir = "debug-unit-tests/non_maximum_suppression/case04_thr";
    fs::create_directories(dir);

    auto g = make_empty_gradients(5, 5);
    g.mag(1, 1) = 9.0f;
    g.mag(2, 2) = 10.0f;
    g.mag(3, 3) = 11.0f;

    const image8u m = libimages::magnitude_threshold_mask(g, 10.0f);
    EXPECT_EQ(m(1, 1), 0);
    EXPECT_EQ(m(2, 2), 255);
    EXPECT_EQ(m(3, 3), 255);

    libimages::debug_io::dump_image((dir / "00_mag.png").string(), g.mag, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask_to_u8_vis(m), true, true);
}
