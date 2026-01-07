#include "draw.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

namespace draw {

namespace {

using libimages::image8u;

static image8u to_rgb_no_alpha(const image8u& img) {
    const int w = img.width();
    const int h = img.height();

    if (img.channels() == 3) return img;

    image8u out(w, h, 3);

    if (img.channels() == 1) {
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                const std::uint8_t v = img(j, i);
                out(j, i, 0) = v;
                out(j, i, 1) = v;
                out(j, i, 2) = v;
            }
        return out;
    }

    rassert(img.channels() == 4, "to_rgb_no_alpha: expected 1/3/4 channels", img.channels());
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            out(j, i, 0) = img(j, i, 0);
            out(j, i, 1) = img(j, i, 1);
            out(j, i, 2) = img(j, i, 2);
        }
    return out;
}

static void darken_inplace_2x(image8u& rgb) {
    rassert(rgb.channels() == 3, "darken_inplace_2x expects RGB", rgb.channels());
    for (int j = 0; j < rgb.height(); ++j)
        for (int i = 0; i < rgb.width(); ++i) {
            rgb(j, i, 0) = static_cast<std::uint8_t>(rgb(j, i, 0) / 2);
            rgb(j, i, 1) = static_cast<std::uint8_t>(rgb(j, i, 1) / 2);
            rgb(j, i, 2) = static_cast<std::uint8_t>(rgb(j, i, 2) / 2);
        }
}

static std::vector<std::array<std::uint8_t, 3>> make_colors(std::size_t n, std::uint32_t seed) {
    FastRandom rng(seed ? seed : 239u);
    std::vector<std::array<std::uint8_t, 3>> cols(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t v = rng.nextU32();
        cols[i] = {
            static_cast<std::uint8_t>(v & 0xFFu),
            static_cast<std::uint8_t>((v >> 8) & 0xFFu),
            static_cast<std::uint8_t>((v >> 16) & 0xFFu),
        };
    }
    return cols;
}

static void put_pixel_thick(image8u& img, int x, int y, const std::array<std::uint8_t, 3>& c, int thickness) {
    const int w = img.width();
    const int h = img.height();
    const int r = std::max(0, thickness - 1);

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const int xx = x + dx;
            const int yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            img(yy, xx, 0) = c[0];
            img(yy, xx, 1) = c[1];
            img(yy, xx, 2) = c[2];
        }
    }
}

static void draw_line_bresenham(image8u& img, point2i a, point2i b, const std::array<std::uint8_t, 3>& c,
                                int thickness) {
    int x0 = a.x, y0 = a.y;
    int x1 = b.x, y1 = b.y;

    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        put_pixel_thick(img, x0, y0, c, thickness);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_circle(image8u& img, point2i center, int radius, const std::array<std::uint8_t, 3>& c,
                        int thickness) {
    radius = std::max(1, radius);
    thickness = std::max(1, thickness);

    const int w = img.width();
    const int h = img.height();

    const int r0 = std::max(0, radius - thickness);
    const int r1 = radius + thickness;
    const int r0_2 = r0 * r0;
    const int r1_2 = r1 * r1;

    for (int dy = -r1; dy <= r1; ++dy) {
        for (int dx = -r1; dx <= r1; ++dx) {
            const int d2 = dx * dx + dy * dy;
            if (d2 < r0_2 || d2 > r1_2) continue;

            const int x = center.x + dx;
            const int y = center.y + dy;
            if (x < 0 || x >= w || y < 0 || y >= h) continue;

            img(y, x, 0) = c[0];
            img(y, x, 1) = c[1];
            img(y, x, 2) = c[2];
        }
    }
}

} // namespace

libimages::image8u overlay_segments_pixels(const image8u& input,
                                          const std::vector<std::vector<point2i>>& segments,
                                          std::uint32_t seed) {
    image8u out = to_rgb_no_alpha(input);
    darken_inplace_2x(out);

    const auto cols = make_colors(segments.size(), seed);

    for (std::size_t si = 0; si < segments.size(); ++si) {
        const auto col = cols[si];
        for (const auto& p : segments[si]) {
            if (p.x < 0 || p.x >= out.width() || p.y < 0 || p.y >= out.height()) continue;
            out(p.y, p.x, 0) = col[0];
            out(p.y, p.x, 1) = col[1];
            out(p.y, p.x, 2) = col[2];
        }
    }
    return out;
}

libimages::image8u overlay_simplified_segments(const image8u& input,
                                               const std::vector<std::pair<point2i, point2i>>& segments,
                                               const DrawParams& p) {
    image8u out = to_rgb_no_alpha(input);
    darken_inplace_2x(out);

    const auto cols = make_colors(segments.size(), p.seed);

    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto col = cols[i];
        draw_line_bresenham(out, segments[i].first, segments[i].second, col, p.line_thickness);
        draw_circle(out, segments[i].first, p.start_circle_radius, col, p.start_circle_thickness);
    }

    return out;
}

void draw_nn_connections_red_inplace(image8u& rgb_overlay,
                                     const std::vector<std::pair<point2i, point2i>>& segments,
                                     const std::vector<int>& nn_B_to_A,
                                     const std::vector<int>& nn_A_to_B,
                                     int thickness) {
    rassert(rgb_overlay.channels() == 3, "draw_nn_connections_red_inplace expects RGB", rgb_overlay.channels());
    rassert(nn_B_to_A.size() == segments.size(), "nn_B_to_A size mismatch", nn_B_to_A.size(), segments.size());
    rassert(nn_A_to_B.size() == segments.size(), "nn_A_to_B size mismatch", nn_A_to_B.size(), segments.size());

    const std::array<std::uint8_t, 3> red = {255, 0, 0};

    // B -> nearest A
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const int j = nn_B_to_A[i];
        if (j < 0) continue;
        draw_line_bresenham(rgb_overlay, segments[i].second, segments[j].first, red, thickness);
    }

    // A -> nearest B
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const int j = nn_A_to_B[i];
        if (j < 0) continue;
        draw_line_bresenham(rgb_overlay, segments[i].first, segments[j].second, red, thickness);
    }
}

libimages::image8u mask_points(const int w, const int h, const std::vector<point2i>& pts) {
    libimages::image8u m(w, h, 1);
    m.fill(static_cast<std::uint8_t>(0));
    for (const auto& p : pts) {
        if (p.x < 0 || p.x >= w || p.y < 0 || p.y >= h) continue;
        m(p.y, p.x) = static_cast<std::uint8_t>(255);
    }
    return m;
}

} // namespace draw
