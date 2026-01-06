#include "downscale.h"

#include <algorithm>
#include <cmath>

#include <libbase/runtime_assert.h>

namespace libimages {

float normalize_downscale_ratio(float r) {
    rassert(r > 0.0f, "kDownscaleRatio must be > 0", r);
    if (r > 1.0f)
        r = 1.0f / r;
    rassert(r > 0.0f && r <= 1.0f, "normalized ratio must be in (0,1]", r);
    return r;
}

image8u downscale_nearest(const image8u &img, float downscale_ratio) {
    const float s = normalize_downscale_ratio(downscale_ratio);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

    if (std::fabs(s - 1.0f) < 1e-7f)
        return img;

    const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
    const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

    image8u out(nw, nh, c);

    const float inv_s = 1.0f / s;
    for (int j = 0; j < nh; ++j) {
        const int sj = std::clamp(static_cast<int>(std::floor((static_cast<float>(j) + 0.5f) * inv_s)), 0, h - 1);
        for (int i = 0; i < nw; ++i) {
            const int si = std::clamp(static_cast<int>(std::floor((static_cast<float>(i) + 0.5f) * inv_s)), 0, w - 1);
            if (c == 1) {
                out(j, i) = img(sj, si);
            } else {
                for (int k = 0; k < c; ++k)
                    out(j, i, k) = img(sj, si, k);
            }
        }
    }
    return out;
}

image8u downscale_bilinear(const image8u &img, float downscale_ratio) {
    const float s = normalize_downscale_ratio(downscale_ratio);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

    if (std::fabs(s - 1.0f) < 1e-7f)
        return img;

    const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
    const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

    image8u out(nw, nh, c);

    const float inv_s = 1.0f / s;

    struct Map1D {
        int i0;
        int i1;
        float t;
    };

    // Precompute X mapping (output x -> source x0/x1 and frac t)
    std::vector<Map1D> xmap(nw);
    for (int i = 0; i < nw; ++i) {
        // Center-aligned mapping: src = (dst+0.5)/s - 0.5
        const float sx = (static_cast<float>(i) + 0.5f) * inv_s - 0.5f;
        const int x0 = static_cast<int>(std::floor(sx));
        const int x1 = x0 + 1;
        const float tx = sx - static_cast<float>(x0);

        xmap[i].i0 = std::clamp(x0, 0, w - 1);
        xmap[i].i1 = std::clamp(x1, 0, w - 1);
        xmap[i].t = std::clamp(tx, 0.0f, 1.0f);
    }

    // Precompute Y mapping
    std::vector<Map1D> ymap(nh);
    for (int j = 0; j < nh; ++j) {
        const float sy = (static_cast<float>(j) + 0.5f) * inv_s - 0.5f;
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = y0 + 1;
        const float ty = sy - static_cast<float>(y0);

        ymap[j].i0 = std::clamp(y0, 0, h - 1);
        ymap[j].i1 = std::clamp(y1, 0, h - 1);
        ymap[j].t = std::clamp(ty, 0.0f, 1.0f);
    }

    auto to_u8 = [](float v) -> uint8_t {
        v = std::clamp(v, 0.0f, 255.0f);
        return static_cast<uint8_t>(std::lround(v));
    };

    for (int j = 0; j < nh; ++j) {
        const int y0 = ymap[j].i0;
        const int y1 = ymap[j].i1;
        const float ty = ymap[j].t;
        const float wy0 = 1.0f - ty;
        const float wy1 = ty;

        for (int i = 0; i < nw; ++i) {
            const int x0 = xmap[i].i0;
            const int x1 = xmap[i].i1;
            const float tx = xmap[i].t;
            const float wx0 = 1.0f - tx;
            const float wx1 = tx;

            if (c == 1) {
                const float v00 = static_cast<float>(img(y0, x0));
                const float v10 = static_cast<float>(img(y0, x1));
                const float v01 = static_cast<float>(img(y1, x0));
                const float v11 = static_cast<float>(img(y1, x1));

                const float v0 = wx0 * v00 + wx1 * v10;
                const float v1 = wx0 * v01 + wx1 * v11;
                out(j, i) = to_u8(wy0 * v0 + wy1 * v1);
            } else {
                for (int k = 0; k < c; ++k) {
                    const float v00 = static_cast<float>(img(y0, x0, k));
                    const float v10 = static_cast<float>(img(y0, x1, k));
                    const float v01 = static_cast<float>(img(y1, x0, k));
                    const float v11 = static_cast<float>(img(y1, x1, k));

                    const float v0 = wx0 * v00 + wx1 * v10;
                    const float v1 = wx0 * v01 + wx1 * v11;
                    out(j, i, k) = to_u8(wy0 * v0 + wy1 * v1);
                }
            }
        }
    }

    return out;
}

image8u downscale_area(const image8u &img, float downscale_ratio) {
    const float s = normalize_downscale_ratio(downscale_ratio);
    if (std::fabs(s - 1.0f) < 1e-7f) return img;

    const float inv_s = 1.0f / s;
    if (inv_s >= 2.0f) {
        // Use area/box filter for strong downscales to reduce aliasing.
        // Threshold can be tuned; inv_s >= 2 means scale-down 2x or more.
        const int w = img.width();
        const int h = img.height();
        const int c = img.channels();
        rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

        const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
        const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

        image8u out(nw, nh, c);

        auto to_u8 = [](double v) -> uint8_t {
            v = std::clamp(v, 0.0, 255.0);
            return static_cast<uint8_t>(std::llround(v));
        };

        struct Span1D {
            int i0;      // inclusive
            int i1;      // inclusive
            float left;  // src coord of left edge (clamped)
            float right; // src coord of right edge (clamped)
        };

        std::vector<Span1D> xspan(nw);
        std::vector<Span1D> yspan(nh);

        // Precompute source footprint for each output pixel in input pixel-space:
        // dst pixel [i, i+1] maps to src [i*inv_s - 0.5, (i+1)*inv_s - 0.5]
        for (int i = 0; i < nw; ++i) {
            float x0 = static_cast<float>(i) * inv_s - 0.5f;
            float x1 = static_cast<float>(i + 1) * inv_s - 0.5f;
            if (x1 < x0) std::swap(x0, x1);

            // Clamp to image extent [0, w]
            x0 = std::clamp(x0, 0.0f, static_cast<float>(w));
            x1 = std::clamp(x1, 0.0f, static_cast<float>(w));

            int ix0 = static_cast<int>(std::floor(x0));
            int ix1 = static_cast<int>(std::ceil(x1)) - 1;

            ix0 = std::clamp(ix0, 0, w - 1);
            ix1 = std::clamp(ix1, 0, w - 1);

            xspan[i] = {ix0, ix1, x0, x1};
        }

        for (int j = 0; j < nh; ++j) {
            float y0 = static_cast<float>(j) * inv_s - 0.5f;
            float y1 = static_cast<float>(j + 1) * inv_s - 0.5f;
            if (y1 < y0) std::swap(y0, y1);

            // Clamp to image extent [0, h]
            y0 = std::clamp(y0, 0.0f, static_cast<float>(h));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(h));

            int iy0 = static_cast<int>(std::floor(y0));
            int iy1 = static_cast<int>(std::ceil(y1)) - 1;

            iy0 = std::clamp(iy0, 0, h - 1);
            iy1 = std::clamp(iy1, 0, h - 1);

            yspan[j] = {iy0, iy1, y0, y1};
        }

        for (int j = 0; j < nh; ++j) {
            const auto ys = yspan[j];
            const double box_h = static_cast<double>(ys.right - ys.left);

            for (int i = 0; i < nw; ++i) {
                const auto xs = xspan[i];
                const double box_w = static_cast<double>(xs.right - xs.left);

                // Degenerate footprint (can happen at extreme edges with tiny output)
                const double box_area = box_w * box_h;
                if (box_area <= 1e-12) {
                    // Fallback to nearest
                    const int si = std::clamp(static_cast<int>(std::floor((static_cast<float>(i) + 0.5f) * inv_s)), 0, w - 1);
                    const int sj = std::clamp(static_cast<int>(std::floor((static_cast<float>(j) + 0.5f) * inv_s)), 0, h - 1);
                    if (c == 1) out(j, i) = img(sj, si);
                    else for (int k = 0; k < c; ++k) out(j, i, k) = img(sj, si, k);
                    continue;
                }

                if (c == 1) {
                    double acc = 0.0;
                    double wsum = 0.0;

                    for (int sy = ys.i0; sy <= ys.i1; ++sy) {
                        const float py0 = static_cast<float>(sy);
                        const float py1 = py0 + 1.0f;
                        const float oy = std::min(ys.right, py1) - std::max(ys.left, py0);
                        if (oy <= 0.0f) continue;

                        for (int sx = xs.i0; sx <= xs.i1; ++sx) {
                            const float px0 = static_cast<float>(sx);
                            const float px1 = px0 + 1.0f;
                            const float ox = std::min(xs.right, px1) - std::max(xs.left, px0);
                            if (ox <= 0.0f) continue;

                            const double wgt = static_cast<double>(ox) * static_cast<double>(oy);
                            acc += wgt * static_cast<double>(img(sy, sx));
                            wsum += wgt;
                        }
                    }

                    out(j, i) = to_u8(acc / (wsum > 0.0 ? wsum : box_area));
                } else {
                    std::vector<double> acc(c, 0.0);
                    double wsum = 0.0;

                    for (int sy = ys.i0; sy <= ys.i1; ++sy) {
                        const float py0 = static_cast<float>(sy);
                        const float py1 = py0 + 1.0f;
                        const float oy = std::min(ys.right, py1) - std::max(ys.left, py0);
                        if (oy <= 0.0f) continue;

                        for (int sx = xs.i0; sx <= xs.i1; ++sx) {
                            const float px0 = static_cast<float>(sx);
                            const float px1 = px0 + 1.0f;
                            const float ox = std::min(xs.right, px1) - std::max(xs.left, px0);
                            if (ox <= 0.0f) continue;

                            const double wgt = static_cast<double>(ox) * static_cast<double>(oy);
                            for (int k = 0; k < c; ++k) {
                                acc[k] += wgt * static_cast<double>(img(sy, sx, k));
                            }
                            wsum += wgt;
                        }
                    }

                    const double denom = (wsum > 0.0 ? wsum : box_area);
                    for (int k = 0; k < c; ++k) out(j, i, k) = to_u8(acc[k] / denom);
                }
            }
        }

        return out;
    } else {
        // Bilinear for mild downscales.
        return downscale_bilinear(img, downscale_ratio);
    }
}

} // namespace libimages
