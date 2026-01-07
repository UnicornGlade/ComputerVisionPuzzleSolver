#include "non_maximum_suppression.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <libbase/point2.h>
#include <libbase/runtime_assert.h>

namespace libimages {

namespace {

// 8-neighborhood offsets
std::array<point2i, 8> kN8 = {
    point2i(-1, -1), point2i(-1, 0), point2i(-1, 1),
    point2i(0, -1),                       point2i(0, 1),
    point2i(1, -1), point2i(1, 0),  point2i(1, 1),
};

static void pick_two_neighbors_from_dir(float dx, float dy, int* out_dj_pos, int* out_di_pos,
                                        int* out_dj_neg, int* out_di_neg) {
    // Pick neighbor with maximum dot(v, dir) and minimum dot(v, dir) over 8 neighbors.
    // dir = (dx, dy) in image coords: x -> columns (i), y -> rows (j)
    float best_pos = -std::numeric_limits<float>::infinity();
    float best_neg = +std::numeric_limits<float>::infinity();

    int dj_pos = 0, di_pos = 0;
    int dj_neg = 0, di_neg = 0;

    for (auto dij : kN8) {
        int di = dij.x;
        int dj = dij.y;

        point2f dxy = dij.normalized();
        const float dot = static_cast<float>(dxy.x) * dx + static_cast<float>(dxy.y) * dy;
        if (dot > best_pos) {
            best_pos = dot;
            dj_pos = dj;
            di_pos = di;
        }
        if (dot < best_neg) {
            best_neg = dot;
            dj_neg = dj;
            di_neg = di;
        }
    }

    *out_dj_pos = dj_pos;
    *out_di_pos = di_pos;
    *out_dj_neg = dj_neg;
    *out_di_neg = di_neg;
}

} // namespace

image8u non_maximum_suppression(const Gradients& g) {
    rassert(g.mag.channels() == 1 && g.dx.channels() == 1 && g.dy.channels() == 1, "Gradients must be 1-channel");
    rassert(g.mag.width() == g.dx.width() && g.mag.height() == g.dx.height(), "Gradients size mismatch (mag/dx)");
    rassert(g.mag.width() == g.dy.width() && g.mag.height() == g.dy.height(), "Gradients size mismatch (mag/dy)");

    const int w = g.mag.width();
    const int h = g.mag.height();

    image8u mask(w, h, 1);
    mask.fill(static_cast<std::uint8_t>(0));

    if (w < 3 || h < 3)
        return mask;

    // Border stays 0.
    for (int j = 1; j < h - 1; ++j) {
        for (int i = 1; i < w - 1; ++i) {
            const float m0 = g.mag(j, i);

            // If gradient is near-zero, treat as not an extremum.
            const float dx = g.dx(j, i);
            const float dy = g.dy(j, i);
            if (std::fabs(dx) + std::fabs(dy) < 1e-12f) {
                mask(j, i) = 0;
                continue;
            }

            int dj_pos = 0, di_pos = 0;
            int dj_neg = 0, di_neg = 0;
            pick_two_neighbors_from_dir(dx, dy, &dj_pos, &di_pos, &dj_neg, &di_neg);

            const float m_pos = g.mag(j + dj_pos, i + di_pos);
            const float m_neg = g.mag(j + dj_neg, i + di_neg);

            // Strict local maximum requirement:
            // If equal or less than any neighbor -> 0.
            mask(j, i) = (m0 > m_pos && m0 > m_neg) ? static_cast<std::uint8_t>(255) : static_cast<std::uint8_t>(0);
        }
    }

    return mask;
}

void apply_mask_inplace(Gradients& g, const image8u& is_ok) {
    rassert(is_ok.channels() == 1, "Mask must be 1-channel", is_ok.channels());
    rassert(is_ok.width() == g.mag.width() && is_ok.height() == g.mag.height(), "Mask size mismatch");

    const int w = g.mag.width();
    const int h = g.mag.height();

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (is_ok(j, i) == 0) {
                g.dx(j, i) = 0.0f;
                g.dy(j, i) = 0.0f;
                g.mag(j, i) = 0.0f;
                if (g.angle.size() != 0) {
                    // Angle may exist; keep consistent.
                    g.angle(j, i) = 0.0f;
                }
            }
        }
    }
}

image8u magnitude_threshold_mask(const Gradients& g, float magnitude_threshold) {
    rassert(g.mag.channels() == 1, "Magnitude must be 1-channel", g.mag.channels());
    rassert(magnitude_threshold >= 0.0f, "magnitude_threshold must be >= 0", magnitude_threshold);

    const int w = g.mag.width();
    const int h = g.mag.height();

    image8u m(w, h, 1);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            m(j, i) = (g.mag(j, i) >= magnitude_threshold) ? static_cast<std::uint8_t>(255) : static_cast<std::uint8_t>(0);
        }
    }
    return m;
}

std::vector<float> extract_non_zero_magnitudes(const Gradients& g) {
    std::vector<float> non_zero_magnitudes;
    for (int j = 0; j < g.mag.height(); ++j) {
        for (int i = 0; i < g.mag.width(); ++i) {
            if (g.mag(j, i) != 0) {
                non_zero_magnitudes.push_back(g.mag(j, i));
            }
        }
    }
    return non_zero_magnitudes;
}

} // namespace libimages
