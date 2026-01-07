#include "morphology.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_black(int w, int h) {
    image8u img(w, h, 1);
    img.fill(static_cast<std::uint8_t>(0));
    return img;
}

static void draw_filled_rect(image8u& img, int x0, int y0, int x1, int y1, std::uint8_t v) {
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

TEST(morphology, SquareErodeDilate_R2) {
    const fs::path dir = "debug-unit-tests/morphology/case00_square";
    fs::create_directories(dir);

    image8u in = make_black(32, 32);
    draw_filled_rect(in, 8, 8, 23, 23, 255);

    const auto er = morphology::erode(in, 2);
    const auto di = morphology::dilate(in, 2);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);
    libimages::debug_io::dump_image((dir / "01_erode_r2.png").string(), er, true, true);
    libimages::debug_io::dump_image((dir / "02_dilate_r2.png").string(), di, true, true);

    // Erosion shrinks: corner near original boundary becomes 0, deep inside stays 255.
    EXPECT_EQ(er(9, 9), 0);
    EXPECT_EQ(er(11, 11), 255);

    // Dilation expands: outside original square becomes 255 near it.
    EXPECT_EQ(di(6, 6), 255);
    EXPECT_EQ(di(5, 5), 0); // with r=2, (5,5) is distance 3 from (8,8) corner => stays 0
}

TEST(morphology, SinglePixel_ErodeAndDilate) {
    const fs::path dir = "debug-unit-tests/morphology/case01_single_pixel";
    fs::create_directories(dir);

    image8u in = make_black(25, 25);
    in(12, 12) = 255;

    const auto er = morphology::erode(in, 1);
    const auto di = morphology::dilate(in, 2);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);
    libimages::debug_io::dump_image((dir / "01_erode_r1.png").string(), er, true, true);
    libimages::debug_io::dump_image((dir / "02_dilate_r2.png").string(), di, true, true);

    EXPECT_EQ(count_white(in), 1);
    EXPECT_EQ(count_white(er), 0);

    // Dilation with radius 2 -> (2r+1)^2 = 25 pixels
    EXPECT_EQ(count_white(di), 25);
    EXPECT_EQ(di(10, 10), 255);
    EXPECT_EQ(di(14, 14), 255);
    EXPECT_EQ(di(9, 9), 0);
}

TEST(morphology, StrengthZero_IsCopy) {
    const fs::path dir = "debug-unit-tests/morphology/case02_strength0";
    fs::create_directories(dir);

    image8u in = make_black(16, 16);
    draw_filled_rect(in, 3, 5, 10, 12, 255);

    const auto er0 = morphology::erode(in, 0);
    const auto di0 = morphology::dilate(in, 0);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), in, true, true);
    libimages::debug_io::dump_image((dir / "01_erode_r0.png").string(), er0, true, true);
    libimages::debug_io::dump_image((dir / "02_dilate_r0.png").string(), di0, true, true);

    EXPECT_EQ(count_white(er0), count_white(in));
    EXPECT_EQ(count_white(di0), count_white(in));
    EXPECT_EQ(er0(7, 7), in(7, 7));
    EXPECT_EQ(di0(7, 7), in(7, 7));
}
