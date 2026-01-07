#include "utils.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_rgb(int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            img(j, i, 0) = r;
            img(j, i, 1) = g;
            img(j, i, 2) = b;
        }
    return img;
}

static image8u make_mask(int w, int h) {
    image8u m(w, h, 1);
    m.fill(static_cast<std::uint8_t>(0));
    return m;
}

static void draw_rect_mask(image8u& m, int x0, int y0, int x1, int y1) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i)
            m(j, i) = 255;
}

static void draw_rect_rgb(image8u& img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i) {
            img(j, i, 0) = r;
            img(j, i, 1) = g;
            img(j, i, 2) = b;
        }
}

TEST(utils_extract, TwoComponents_8Connected) {
    const fs::path dir = "debug-unit-tests/utils/extract_objects/case00";
    fs::create_directories(dir);

    image8u img = make_rgb(120, 90, 30, 30, 30);
    image8u mask = make_mask(120, 90);

    // Component A: red block
    draw_rect_mask(mask, 10, 10, 35, 28);
    draw_rect_rgb(img, 10, 10, 35, 28, 255, 0, 0);

    // Component B: green block
    draw_rect_mask(mask, 70, 40, 105, 75);
    draw_rect_rgb(img, 70, 40, 105, 75, 0, 255, 0);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask, true, true);

    utils::ExtractParams p;
    p.eight_connected = true;

    const auto objs = utils::extract_objects_by_mask<std::uint8_t>(img, mask, p);
    ASSERT_EQ(objs.size(), 2u);

    // Deterministic order: by top-left (y,x)
    EXPECT_EQ(objs[0].offset.x, 10);
    EXPECT_EQ(objs[0].offset.y, 10);

    EXPECT_EQ(objs[1].offset.x, 70);
    EXPECT_EQ(objs[1].offset.y, 40);

    for (std::size_t k = 0; k < objs.size(); ++k) {
        const auto& o = objs[k];
        const std::string base = "obj" + std::to_string(k);

        libimages::debug_io::dump_image((dir / (base + "_image.png")).string(), o.image, true, true);
        libimages::debug_io::dump_image((dir / (base + "_mask.png")).string(), o.mask, true, true);

        // Basic sanity: mask top-left should be 255 for these solid rectangles.
        EXPECT_EQ(o.mask(0, 0), 255);
    }
}

TEST(utils_extract, DiagonalConnectivity_4vs8) {
    const fs::path dir = "debug-unit-tests/utils/extract_objects/case01_diag";
    fs::create_directories(dir);

    image8u img = make_rgb(40, 40, 0, 0, 0);
    image8u mask = make_mask(40, 40);

    // Two pixels touching only diagonally
    mask(10, 10) = 255;
    mask(11, 11) = 255;
    img(10, 10, 0) = 255;
    img(11, 11, 1) = 255;

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), mask, true, true);

    {
        utils::ExtractParams p;
        p.eight_connected = false;
        const auto objs = utils::extract_objects_by_mask<std::uint8_t>(img, mask, p);
        EXPECT_EQ(objs.size(), 2u);
        for (std::size_t k = 0; k < objs.size(); ++k) {
            libimages::debug_io::dump_image((dir / ("04conn_obj" + std::to_string(k) + "_mask.png")).string(),
                                            objs[k].mask, true, true);
        }
    }

    {
        utils::ExtractParams p;
        p.eight_connected = true;
        const auto objs = utils::extract_objects_by_mask<std::uint8_t>(img, mask, p);
        EXPECT_EQ(objs.size(), 1u);
        libimages::debug_io::dump_image((dir / "08conn_obj0_mask.png").string(), objs[0].mask, true, true);
    }
}
