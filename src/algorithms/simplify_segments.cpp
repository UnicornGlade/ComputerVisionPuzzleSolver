#include "simplify_segments.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

namespace simplify_segments {

namespace {

constexpr float kPi = 3.14159265358979323846f;

static float deg2rad(float d) { return d * kPi / 180.0f; }

static libimages::image8u to_rgb_no_alpha(const libimages::image8u& img) {
    const int w = img.width();
    const int h = img.height();

    if (img.channels() == 3) return img;

    libimages::image8u out(w, h, 3);

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

    rassert(img.channels() == 4, "Unexpected channel count", img.channels());
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            out(j, i, 0) = img(j, i, 0);
            out(j, i, 1) = img(j, i, 1);
            out(j, i, 2) = img(j, i, 2);
        }
    return out;
}

static void darken_inplace_2x(libimages::image8u& img) {
    rassert(img.channels() == 3, "darken_inplace_2x expects RGB", img.channels());
    for (int j = 0; j < img.height(); ++j)
        for (int i = 0; i < img.width(); ++i) {
            img(j, i, 0) = static_cast<std::uint8_t>(img(j, i, 0) / 2);
            img(j, i, 1) = static_cast<std::uint8_t>(img(j, i, 1) / 2);
            img(j, i, 2) = static_cast<std::uint8_t>(img(j, i, 2) / 2);
        }
}

static void put_pixel_thick(libimages::image8u& img, int x, int y,
                            const std::array<std::uint8_t, 3>& c, int thickness) {
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

static void draw_line_bresenham(libimages::image8u& img, point2i a, point2i b,
                                const std::array<std::uint8_t, 3>& c, int thickness) {
    rassert(img.channels() == 3, "draw_line_bresenham expects RGB", img.channels());

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

} // namespace

Segment simplify_segment(const find_segments::SegmentPixels& s) {
    rassert(!s.pixels.empty(), "simplify_segment: empty pixels");

    // 1) Centroid of pixel centers.
    double sx = 0.0;
    double sy = 0.0;
    for (const auto& p : s.pixels) {
        sx += static_cast<double>(p.x) + 0.5;
        sy += static_cast<double>(p.y) + 0.5;
    }
    const double cx = sx / static_cast<double>(s.pixels.size());
    const double cy = sy / static_cast<double>(s.pixels.size());

    // 2) Line direction through centroid, perpendicular to median_angle_deg (gradient).
    // Edge direction = grad + 90 deg.
    const float ang = deg2rad(s.median_angle_deg + 90.0f);
    const float ux = std::cos(ang);
    const float uy = std::sin(ang);

    // 3..4) A = min projection, B = max projection.
    double best_min = std::numeric_limits<double>::infinity();
    double best_max = -std::numeric_limits<double>::infinity();
    point2i best_a = s.pixels[0];
    point2i best_b = s.pixels[0];

    for (const auto& p : s.pixels) {
        const double x = static_cast<double>(p.x) + 0.5;
        const double y = static_cast<double>(p.y) + 0.5;
        const double px = x - cx;
        const double py = y - cy;
        const double t = px * static_cast<double>(ux) + py * static_cast<double>(uy);

        if (t < best_min) { best_min = t; best_a = p; }
        if (t > best_max) { best_max = t; best_b = p; }
    }

    return Segment{best_a, best_b};
}

std::vector<Segment> simplify_segments(const std::vector<find_segments::SegmentPixels>& segments) {
    std::vector<Segment> out;
    out.reserve(segments.size());
    for (const auto& s : segments) out.push_back(simplify_segment(s));
    return out;
}

libimages::image8u visualize_segments_overlay(const libimages::image8u& input,
                                             const std::vector<Segment>& segments,
                                             const VisualizeParams& vis) {
    libimages::image8u out = to_rgb_no_alpha(input);
    darken_inplace_2x(out);

    FastRandom rng(vis.seed ? vis.seed : 239U);

    for (std::size_t i = 0; i < segments.size(); ++i) {
        const std::uint32_t v = rng.nextU32();
        const std::array<std::uint8_t, 3> col = {
            static_cast<std::uint8_t>(v & 0xFFU),
            static_cast<std::uint8_t>((v >> 8) & 0xFFU),
            static_cast<std::uint8_t>((v >> 16) & 0xFFU),
        };
        draw_line_bresenham(out, segments[i].a, segments[i].b, col, vis.thickness);
    }

    return out;
}

} // namespace simplify_segments
