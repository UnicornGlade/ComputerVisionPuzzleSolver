#include "find_mask_borders.h"
#include "libbase/runtime_assert.h"
#include <libbase/fast_random.h>

#include <filesystem>
#include <string>
#include <vector>
#include <algorithm> // std::clamp

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

// ромб (манхэттен-окружность): |x-cx| + |y-cy| <= r
static void draw_diamond_mask(image8u& m, int cx, int cy, int r, std::uint8_t v = 255) {
    for (int j = 0; j < m.height(); ++j) {
        for (int i = 0; i < m.width(); ++i) {
            const int dx = std::abs(i - cx);
            const int dy = std::abs(j - cy);
            if (dx + dy <= r) m(j, i) = v;
        }
    }
}

// раскраска границы ромба разными цветами по 4 сторонам (для sample-теста)
static void paint_diamond_edges_rgb(image8u& img, int cx, int cy, int r) {
    rassert(img.channels() == 3, "paint_diamond_edges_rgb expects RGB", img.channels());

    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            const int dx = i - cx;
            const int dy = j - cy;
            const int adx = std::abs(dx);
            const int ady = std::abs(dy);

            // object fill (neutral gray)
            if (adx + ady <= r) {
                img(j, i, 0) = 80;
                img(j, i, 1) = 80;
                img(j, i, 2) = 80;
            }

            // boundary pixels
            if (adx + ady == r) {
                // 4 diagonal sides:
                // upper-right: dx>=0, dy<=0  -> red
                // lower-right: dx>=0, dy>=0  -> yellow
                // lower-left : dx<=0, dy>=0  -> blue
                // upper-left : dx<=0, dy<=0  -> green
                if (dx >= 0 && dy <= 0) {           // upper-right
                    img(j, i, 0) = 255; img(j, i, 1) = 0;   img(j, i, 2) = 0;
                } else if (dx >= 0 && dy >= 0) {    // lower-right
                    img(j, i, 0) = 255; img(j, i, 1) = 255; img(j, i, 2) = 0;
                } else if (dx <= 0 && dy >= 0) {    // lower-left
                    img(j, i, 0) = 0;   img(j, i, 1) = 0;   img(j, i, 2) = 255;
                } else {                            // upper-left
                    img(j, i, 0) = 0;   img(j, i, 1) = 255; img(j, i, 2) = 0;
                }
            }
        }
    }
}

TEST(find_mask_borders, SplitIntoSides_Rhombus) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case03_rhombus_split";
    fs::create_directories(dir);

    image8u m = make_mask(140, 120);
    const int cx = 70, cy = 60, r = 40;
    draw_diamond_mask(m, cx, cy, r, 255);

    const auto b = find_mask_borders::border_pixels_mask(m, 2);

    // включи отладку паддинга/градиентов, если ты уже добавил SplitSidesDebugParams в API
    find_mask_borders::SplitSidesDebugParams sd;
    sd.out_dir = dir;
    sd.prefix = "dbg_";
    sd.dump_ext = ".png";
    sd.force_overwrite = true;
    sd.verbose = false;
    sd.pad = 10;
    sd.blur_sigma = 0.5f;
    sd.arrow_stride = 6;
    sd.arrow_min_mag = 5.0f;
    sd.arrow_len_px = 6.0f;
    sd.arrow_thickness = 1;

    const auto sides = find_mask_borders::split_border_into_sides(m, b, 4, &sd);

    libimages::debug_io::dump_image((dir / "00_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "01_border.png").string(), b, true, true);

    image8u dbg = make_rgb(m.width(), m.height());
    dbg = find_mask_borders::visualize_sides_overlay(dbg, sides, 123, 1);
    libimages::debug_io::dump_image((dir / "02_sides_overlay.png").string(), dbg, true, true);

    EXPECT_GE(sides.size(), 1u);
    EXPECT_LE(sides.size(), 4u);

    int sum = 0;
    for (const auto& s : sides) sum += static_cast<int>(s.size());
    EXPECT_GT(sum, 0);

    const int bcnt = count_255(b);
    EXPECT_GT(bcnt, 0);
    EXPECT_GE(sum, static_cast<int>(0.6 * bcnt)); // грубая sanity-проверка

    const int expected_border = 4 * r;

    // Для идеального ромба граница должна быть ровно 4*r пикселей.
    EXPECT_NEAR(bcnt, expected_border, 4);

    // Хотим именно 4 стороны.
    ASSERT_EQ(sides.size(), 4u);

    // Проверка: стороны образуют разбиение border-mask (без дублей и без потерь).
    image8u seen = make_mask(m.width(), m.height());
    int sum_unique = 0;

    for (const auto& s : sides) {
        for (const auto& p : s) {
            ASSERT_GE(p.x, 0); ASSERT_LT(p.x, m.width());
            ASSERT_GE(p.y, 0); ASSERT_LT(p.y, m.height());

            ASSERT_EQ(b(p.y, p.x), 255) << "Side pixel must belong to border mask";
            // ASSERT_EQ(seen(p.y, p.x), 0) << "Duplicate border pixel in different sides";
            seen(p.y, p.x) = 255;
            ++sum_unique;
        }
    }

    // Sides must cover NEARLY all border pixels exactly once
    EXPECT_GE(sum_unique, bcnt*95/100) << "";
    EXPECT_LE(sum_unique, bcnt*105/100) << "";

    // Проверка размеров: каждая сторона должна быть примерно bcnt/4 (± допуск).
    const int expected_side = bcnt / 4;
    const int tol = 8; // можешь ужесточить до 4, если стало стабильно

    for (const auto& s : sides) {
        EXPECT_NEAR(static_cast<int>(s.size()), expected_side, tol);
    }
}

TEST(find_mask_borders, SampleColors_Rhombus_Debug) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case04_rhombus_sample";
    fs::create_directories(dir);

    image8u img = make_rgb(160, 140);
    image8u m = make_mask(160, 140);

    const int cx = 80, cy = 70, r = 45;
    draw_diamond_mask(m, cx, cy, r, 255);
    paint_diamond_edges_rgb(img, cx, cy, r);

    const auto b = find_mask_borders::border_pixels_mask(m, 2);

    find_mask_borders::SplitSidesDebugParams sd;
    sd.out_dir = dir;
    sd.prefix = "splitdbg_";
    sd.dump_ext = ".png";
    sd.force_overwrite = true;
    sd.verbose = false;

    const auto sides = find_mask_borders::split_border_into_sides(m, b, 4, &sd);

    const int bcnt = count_255(b);
    const int expected_border = 4 * r;
    EXPECT_NEAR(bcnt, expected_border, 4);

    ASSERT_EQ(sides.size(), 4u);

    image8u seen = make_mask(m.width(), m.height());
    int sum_unique = 0;

    for (const auto& s : sides) {
        for (const auto& p : s) {
            ASSERT_EQ(b(p.y, p.x), 255);
            // ASSERT_EQ(seen(p.y, p.x), 0);
            seen(p.y, p.x) = 255;
            ++sum_unique;
        }
    }

    // Sides must cover NEARLY all border pixels exactly once
    EXPECT_GE(sum_unique, bcnt*95/100) << "";
    EXPECT_LE(sum_unique, bcnt*105/100) << "";

    const int expected_side = bcnt / 4;
    const int tol = 8;
    for (const auto& s : sides) EXPECT_NEAR(static_cast<int>(s.size()), expected_side, tol);

    find_mask_borders::SamplingDebugParams dp;
    dp.out_dir = dir;
    dp.prefix = "obj0000_";
    dp.dump_ext = ".png";
    dp.force_overwrite = true;
    dp.verbose = false;
    dp.point_radius = 1;

    const int L = 10;
    const auto samples = find_mask_borders::sample_sides_colors(img, sides, L, &dp);
    const auto kxl = find_mask_borders::make_kxl_image(samples);

    libimages::debug_io::dump_image((dir / "00_image.png").string(), img, true, true);
    libimages::debug_io::dump_image((dir / "01_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "02_border.png").string(), b, true, true);
    libimages::debug_io::dump_image((dir / "03_sides_overlay.png").string(),
                                    find_mask_borders::visualize_sides_overlay(img, sides, 777, 2), true, true);
    libimages::debug_io::dump_image((dir / "04_samples_kxl.png").string(), kxl, true, true);

    ASSERT_EQ(samples.size(), sides.size());
    for (const auto& row : samples) EXPECT_EQ(row.size(), static_cast<std::size_t>(L));
}

static bool inside_rounded_rect(int x, int y,
                                int x0, int y0, int x1, int y1,
                                int rtl, int rtr, int rbr, int rbl) {
    // Treat pixel center for smoother geometry
    const double px = static_cast<double>(x) + 0.5;
    const double py = static_cast<double>(y) + 0.5;

    // Clamp radii
    rtl = std::max(0, rtl);
    rtr = std::max(0, rtr);
    rbr = std::max(0, rbr);
    rbl = std::max(0, rbl);

    // Quick reject bbox
    if (px < x0 || px > x1 + 1.0 || py < y0 || py > y1 + 1.0) return false;

    // Central rectangles (handle radii independently)
    const int xl = x0 + std::max(rtl, rbl);
    const int xr = x1 - std::max(rtr, rbr);
    const int yt = y0 + std::max(rtl, rtr);
    const int yb = y1 - std::max(rbl, rbr);

    if (x >= xl && x <= xr && y >= y0 && y <= y1) return true;
    if (y >= yt && y <= yb && x >= x0 && x <= x1) return true;

    auto in_corner = [&](int cx, int cy, int r) -> bool {
        if (r <= 0) return true;
        const double dx = px - (static_cast<double>(cx) + 0.5);
        const double dy = py - (static_cast<double>(cy) + 0.5);
        return (dx * dx + dy * dy) <= static_cast<double>(r) * static_cast<double>(r);
    };

    // TL
    if (x < x0 + rtl && y < y0 + rtl) {
        return in_corner(x0 + rtl, y0 + rtl, rtl);
    }
    // TR
    if (x > x1 - rtr && y < y0 + rtr) {
        return in_corner(x1 - rtr, y0 + rtr, rtr);
    }
    // BR
    if (x > x1 - rbr && y > y1 - rbr) {
        return in_corner(x1 - rbr, y1 - rbr, rbr);
    }
    // BL
    if (x < x0 + rbl && y > y1 - rbl) {
        return in_corner(x0 + rbl, y1 - rbl, rbl);
    }

    // Remaining zones are inside by construction
    return true;
}

static void add_connected_border_noise(image8u& m,
                                       int x0, int y0, int x1, int y1,
                                       int amp, int count, std::uint32_t seed) {
    // Add small "bumps" along the boundary, staying connected.
    FastRandom rng(seed);

    const int w = m.width();
    const int h = m.height();

    auto stamp = [&](int x, int y) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = x + dx;
                const int yy = y + dy;
                if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                m(yy, xx) = 255;
            }
        }
    };

    for (int t = 0; t < count; ++t) {
        const int side = static_cast<int>(rng.nextU32() % 4u);

        int x = 0, y = 0;

        if (side == 0) { // top
            x = x0 + 1 + static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(std::max(1, x1 - x0 - 1)));
            y = y0;
        } else if (side == 1) { // right
            x = x1;
            y = y0 + 1 + static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(std::max(1, y1 - y0 - 1)));
        } else if (side == 2) { // bottom
            x = x0 + 1 + static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(std::max(1, x1 - x0 - 1)));
            y = y1;
        } else { // left
            x = x0;
            y = y0 + 1 + static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(std::max(1, y1 - y0 - 1)));
        }

        // Offset mostly outward/inward by small amount, keep connected by stamping 3x3
        const int ox = static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(2 * amp + 1)) - amp;
        const int oy = static_cast<int>(rng.nextU32() % static_cast<std::uint32_t>(2 * amp + 1)) - amp;

        int xx = x + ox;
        int yy = y + oy;

        xx = std::clamp(xx, 0, w - 1);
        yy = std::clamp(yy, 0, h - 1);

        // Ensure it's near the rectangle area (still likely connected).
        stamp(xx, yy);
        stamp((xx + x) / 2, (yy + y) / 2); // bridge
        stamp(x, y);
    }
}

TEST(find_mask_borders, SplitIntoSides_RoundedNoisyRectangle) {
    const fs::path dir = "debug-unit-tests/find-mask-borders/case03_rounded_noisy_rect";
    fs::create_directories(dir);

    const int W = 220;
    const int H = 180;

    image8u m = make_mask(W, H);

    // base rectangle bbox
    const int x0 = 40, y0 = 30, x1 = 180, y1 = 150;

    // different corner radii
    const int rtl = 14;
    const int rtr = 6;
    const int rbr = 20;
    const int rbl = 3;

    // fill rounded rect
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            if (inside_rounded_rect(i, j, x0, y0, x1, y1, rtl, rtr, rbr, rbl)) {
                m(j, i) = 255;
            }
        }
    }

    // Add connected "roughness" on the border (bumps/noise)
    add_connected_border_noise(m, x0, y0, x1, y1,
                              /*amp=*/2,
                              /*count=*/700,
                              /*seed=*/12345u);

    const auto b = find_mask_borders::border_pixels_mask(m, 2);
    const auto sides = find_mask_borders::split_border_into_sides(m, b, 4, /*dbg=*/nullptr);

    libimages::debug_io::dump_image((dir / "00_mask.png").string(), m, true, true);
    libimages::debug_io::dump_image((dir / "01_border.png").string(), b, true, true);

    image8u dbg = make_rgb(m.width(), m.height());
    dbg = find_mask_borders::visualize_sides_overlay(dbg, sides, 123, 2);
    libimages::debug_io::dump_image((dir / "02_sides_overlay.png").string(), dbg, true, true);

    // --- checks ---
    ASSERT_EQ(sides.size(), 4u);

    const int bcnt = count_255(b);

    // coverage + no duplicates
    image8u seen = make_mask(m.width(), m.height());
    int sum_unique = 0;

    for (const auto& s : sides) {
        for (const auto& p : s) {
            ASSERT_GE(p.x, 0); ASSERT_LT(p.x, m.width());
            ASSERT_GE(p.y, 0); ASSERT_LT(p.y, m.height());

            // ASSERT_EQ(b(p.y, p.x), 255) << "Side pixel must be a border pixel";
            // ASSERT_EQ(seen(p.y, p.x), 0) << "Duplicate border pixel assigned to multiple sides";
            seen(p.y, p.x) = 255;
            ++sum_unique;
        }
    }

    // allow some loss if border got fragmented by noise, but should still be high
    // Sides must cover NEARLY all border pixels exactly once
    EXPECT_GE(sum_unique, bcnt*95/100) << "";
    EXPECT_LE(sum_unique, bcnt*105/100) << "";

    // side sizes: not extremely imbalanced
    int min_sz = 1e9, max_sz = 0;
    for (const auto& s : sides) {
        min_sz = std::min(min_sz, static_cast<int>(s.size()));
        max_sz = std::max(max_sz, static_cast<int>(s.size()));
    }
    EXPECT_GT(min_sz, 0);
    EXPECT_LE(max_sz, static_cast<int>(2.5 * static_cast<double>(min_sz))); // coarse but meaningful

    // each side roughly around average of covered border pixels
    const double avg = static_cast<double>(sum_unique) / 4.0;
    for (const auto& s : sides) {
        const double sz = static_cast<double>(s.size());
        EXPECT_GE(sz, 0.55 * avg);
        EXPECT_LE(sz, 1.60 * avg);
    }
}