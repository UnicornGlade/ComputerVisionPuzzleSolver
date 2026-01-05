#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

// All tunable parameters live here.
namespace cfg {

// Output naming.
inline constexpr const char *kDefaultDumpExt = ".jpg"; // png is lossless, but jpg is faster

// Debug IO behavior.
inline constexpr bool kDumpVerbose = true;        // true -> print extra info while saving.
inline constexpr bool kDumpForceOverwrite = true; // true -> overwrite existing files.

// Preprocessing.
inline constexpr float kDownscaleRatio = 4.0f; // 1.0 -> no resize. <1 shrinks. >1 is treated as 1/kDownscaleRatio.
inline constexpr float kGaussianSigma =
    0.8f; // Larger -> more smoothing (less noise, fewer edges). Smaller -> sharper but noisier.

// Graph-based gradient clustering (Olson 2011, Eq.(1)).
inline constexpr float kClusterKD =
    100.0f; // Larger -> allow more angular spread, merges more components. Smaller -> splits more.
inline constexpr float kClusterKM =
    1200.0f; // Larger -> allow more magnitude spread, merges more. Smaller -> splits more.

// Segment fitting filters.
inline constexpr int kMinComponentPixels =
    25; // Larger -> fewer segments (discard small components). Smaller -> more tiny segments.
inline constexpr float kMinMeanMagnitude = 10.0f; // Larger -> keep only strong edges. Smaller -> keep weak/noisy edges.
inline constexpr float kMinSegmentLengthPx =
    12.0f; // Larger -> discard short segments. Smaller -> keep short fragments.

// Visualization.
inline constexpr float kNotchLengthPx = 10.0f; // Larger -> longer direction notch, smaller -> shorter.

} // namespace cfg

// NOTE: All comments are in English by request.
namespace fs = std::filesystem;

namespace {

constexpr float kPi = 3.14159265358979323846f;

using libimages::image32f;
using libimages::image8u;

static std::string stage_path(const fs::path &out_dir, int step, const std::string &name,
                              const std::string &ext = cfg::kDefaultDumpExt) {
    const std::string filename = (step < 10 ? "0" : "") + std::to_string(step) + "_" + name + ext;
    return (out_dir / filename).string();
}

static float normalize_downscale_ratio(float r) {
    // Interpret cfg::kDownscaleRatio as:
    // - r == 1.0 -> no resize
    // - 0 < r < 1 -> shrink by factor r
    // - r > 1 -> also shrink, treated as 1/r (so user can pass 2.0 meaning half size)
    rassert(r > 0.0f, "kDownscaleRatio must be > 0", r);
    if (r > 1.0f) r = 1.0f / r;
    rassert(r > 0.0f && r <= 1.0f, "normalized ratio must be in (0,1]", r);
    return r;
}

static image8u downscale_nearest(const image8u& img, float downscale_ratio) {
    // Simple nearest-neighbor downscale for debug/prototyping.
    // Preserves channel count (1/3/4 supported, but works for any c_>0).
    const float s = normalize_downscale_ratio(downscale_ratio);
    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

    if (std::fabs(s - 1.0f) < 1e-7f) {
        return img; // No resize.
    }

    const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
    const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

    image8u out(nw, nh, c);

    const float inv_s = 1.0f / s;
    for (int j = 0; j < nh; ++j) {
        // Map dst pixel center -> src pixel index.
        const int sj = std::clamp(static_cast<int>(std::floor((static_cast<float>(j) + 0.5f) * inv_s)), 0, h - 1);
        for (int i = 0; i < nw; ++i) {
            const int si = std::clamp(static_cast<int>(std::floor((static_cast<float>(i) + 0.5f) * inv_s)), 0, w - 1);
            if (c == 1) {
                out(j, i) = img(sj, si);
            } else {
                for (int k = 0; k < c; ++k) out(j, i, k) = img(sj, si, k);
            }
        }
    }

    return out;
}

static image32f to_grayscale_float(const image8u &img) {
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported channel count",
            img.channels());

    image32f gray(img.width(), img.height(), 1);

    if (img.channels() == 1) {
        for (int j = 0; j < img.height(); ++j) {
            for (int i = 0; i < img.width(); ++i) {
                gray(j, i) = static_cast<float>(img(j, i));
            }
        }
        return gray;
    }

    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            const float r = static_cast<float>(img(j, i, 0));
            const float g = static_cast<float>(img(j, i, 1));
            const float b = static_cast<float>(img(j, i, 2));
            gray(j, i) = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
    return gray;
}

static std::vector<float> gaussian_kernel_1d(float sigma, int *out_radius) {
    rassert(sigma > 0.0f, "sigma must be positive", sigma);

    const int radius = static_cast<int>(std::ceil(3.0f * sigma));
    *out_radius = radius;

    std::vector<float> k(static_cast<std::size_t>(2 * radius + 1));
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    float sum = 0.0f;
    for (int x = -radius; x <= radius; ++x) {
        const float v = std::exp(-static_cast<float>(x * x) * inv2s2);
        k[static_cast<std::size_t>(x + radius)] = v;
        sum += v;
    }

    for (float &v : k)
        v /= sum;
    return k;
}

static image32f gaussian_blur_gray(const image32f &gray, float sigma) {
    rassert(gray.channels() == 1, "gaussian_blur_gray expects grayscale", gray.channels());

    const int w = gray.width();
    const int h = gray.height();

    int radius = 0;
    const std::vector<float> k = gaussian_kernel_1d(sigma, &radius);

    image32f tmp(w, h, 1);
    image32f out(w, h, 1);

    // Horizontal pass (replicate border).
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            float acc = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx) {
                int x = i + dx;
                if (x < 0)
                    x = 0;
                if (x >= w)
                    x = w - 1;
                acc += gray(j, x) * k[static_cast<std::size_t>(dx + radius)];
            }
            tmp(j, i) = acc;
        }
    }

    // Vertical pass (replicate border).
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            float acc = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                int y = j + dy;
                if (y < 0)
                    y = 0;
                if (y >= h)
                    y = h - 1;
                acc += tmp(y, i) * k[static_cast<std::size_t>(dy + radius)];
            }
            out(j, i) = acc;
        }
    }

    return out;
}

struct Gradients {
    image32f dx;    // 1-channel
    image32f dy;    // 1-channel
    image32f mag;   // 1-channel
    image32f angle; // 1-channel, degrees in [0, 360)
};

static float wrap_angle_deg(float a) {
    a = std::fmod(a, 360.0f);
    if (a < 0.0f)
        a += 360.0f;
    return a;
}

static float angle_diff_deg(float a, float b) {
    float d = std::fabs(a - b);
    if (d > 180.0f)
        d = 360.0f - d;
    return d;
}

static Gradients sobel_gradients(const image32f &gray_blurred) {
    rassert(gray_blurred.channels() == 1, "sobel_gradients expects grayscale", gray_blurred.channels());

    const int w = gray_blurred.width();
    const int h = gray_blurred.height();

    Gradients g{
        image32f(w, h, 1),
        image32f(w, h, 1),
        image32f(w, h, 1),
        image32f(w, h, 1),
    };

    g.dx.fill(0.0f);
    g.dy.fill(0.0f);
    g.mag.fill(0.0f);
    g.angle.fill(0.0f);

    if (w < 3 || h < 3)
        return g;

    for (int j = 1; j < h - 1; ++j) {
        for (int i = 1; i < w - 1; ++i) {
            const float p00 = gray_blurred(j - 1, i - 1);
            const float p01 = gray_blurred(j - 1, i);
            const float p02 = gray_blurred(j - 1, i + 1);

            const float p10 = gray_blurred(j, i - 1);
            const float p12 = gray_blurred(j, i + 1);

            const float p20 = gray_blurred(j + 1, i - 1);
            const float p21 = gray_blurred(j + 1, i);
            const float p22 = gray_blurred(j + 1, i + 1);

            const float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            const float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

            g.dx(j, i) = gx;
            g.dy(j, i) = gy;

            const float m = std::sqrt(gx * gx + gy * gy);
            g.mag(j, i) = m;

            g.angle(j, i) = (m > 1e-6f) ? wrap_angle_deg(std::atan2(gy, gx) * 180.0f / kPi) : 0.0f;
        }
    }

    return g;
}

static image8u visualize_signed_to_u8(const image32f &img) {
    rassert(img.channels() == 1, "visualize_signed_to_u8 expects 1-channel", img.channels());

    const int w = img.width();
    const int h = img.height();

    float max_abs = 0.0f;
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            max_abs = std::max(max_abs, std::fabs(img(j, i)));

    if (max_abs < 1e-6f)
        max_abs = 1.0f;

    image8u out(w, h, 1);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const float v = img(j, i) / max_abs; // [-1, 1]
            int u = static_cast<int>(std::lround(127.5f + 127.5f * v));
            u = std::clamp(u, 0, 255);
            out(j, i) = static_cast<std::uint8_t>(u);
        }
    }
    return out;
}

static void hsv_to_rgb(float h, float s, float v, std::uint8_t *r, std::uint8_t *g, std::uint8_t *b) {
    const float hh = h * 6.0f;
    const int i = static_cast<int>(std::floor(hh));
    const float f = hh - static_cast<float>(i);

    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    switch (i % 6) {
    case 0:
        rr = v;
        gg = t;
        bb = p;
        break;
    case 1:
        rr = q;
        gg = v;
        bb = p;
        break;
    case 2:
        rr = p;
        gg = v;
        bb = t;
        break;
    case 3:
        rr = p;
        gg = q;
        bb = v;
        break;
    case 4:
        rr = t;
        gg = p;
        bb = v;
        break;
    case 5:
        rr = v;
        gg = p;
        bb = q;
        break;
    }

    auto to_u8 = [](float x) -> std::uint8_t {
        int u = static_cast<int>(std::lround(x * 255.0f));
        u = std::clamp(u, 0, 255);
        return static_cast<std::uint8_t>(u);
    };

    *r = to_u8(rr);
    *g = to_u8(gg);
    *b = to_u8(bb);
}

static image8u visualize_angle_hsv(const image32f &angle_deg, const image32f &mag) {
    rassert(angle_deg.channels() == 1 && mag.channels() == 1, "visualize_angle_hsv expects 1-channel inputs");

    const int w = angle_deg.width();
    const int h = angle_deg.height();
    rassert(mag.width() == w && mag.height() == h, "angle/mag size mismatch");

    float max_mag = 0.0f;
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            max_mag = std::max(max_mag, mag(j, i));
    if (max_mag < 1e-6f)
        max_mag = 1.0f;

    image8u out(w, h, 3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const float a = wrap_angle_deg(angle_deg(j, i));
            const float h01 = a / 360.0f;
            const float v01 = std::clamp(mag(j, i) / max_mag, 0.0f, 1.0f);

            std::uint8_t r, g, b;
            hsv_to_rgb(h01, 1.0f, v01, &r, &g, &b);
            out(j, i, 0) = r;
            out(j, i, 1) = g;
            out(j, i, 2) = b;
        }
    }
    return out;
}

struct UnionFind {
    int w = 0;
    int h = 0;

    std::vector<int> parent;
    std::vector<int> size;

    std::vector<float> min_mag;
    std::vector<float> max_mag;

    // Angle stats in degrees, stored in an "unwrapped" space (can exceed [0,360)).
    std::vector<float> min_ang;
    std::vector<float> max_ang;

    explicit UnionFind(int width, int height) : w(width), h(height) {
        const int n = w * h;
        parent.resize(n);
        size.assign(n, 1);
        min_mag.resize(n);
        max_mag.resize(n);
        min_ang.resize(n);
        max_ang.resize(n);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    int find(int x) const {
        while (parent[x] != x)
            x = parent[x];
        return x;
    }

    static float center(float mn, float mx) { return 0.5f * (mn + mx); }

    // Align component B to component A by shifting by multiples of 360 degrees.
    static float compute_shift_360(float centerA, float centerB) {
        const float k = std::round((centerA - centerB) / 360.0f);
        return k * 360.0f;
    }

    bool can_merge(int ra, int rb, float KD, float KM) const {
        if (ra == rb)
            return false;

        const int sa = size[ra];
        const int sb = size[rb];
        const int s = sa + sb;

        const float Da = max_ang[ra] - min_ang[ra];
        const float Db = max_ang[rb] - min_ang[rb];

        const float Ma = max_mag[ra] - min_mag[ra];
        const float Mb = max_mag[rb] - min_mag[rb];

        const float ca = center(min_ang[ra], max_ang[ra]);
        const float cb = center(min_ang[rb], max_ang[rb]);
        const float sh = compute_shift_360(ca, cb);

        const float minB = min_ang[rb] + sh;
        const float maxB = max_ang[rb] + sh;

        const float Dunion = std::max(max_ang[ra], maxB) - std::min(min_ang[ra], minB);
        const float Munion = std::max(max_mag[ra], max_mag[rb]) - std::min(min_mag[ra], min_mag[rb]);

        // Olson 2011, Eq. (1):
        // D(n ∪ m) ≤ min(D(n), D(m)) + KD / |n ∪ m|
        // M(n ∪ m) ≤ min(M(n), M(m)) + KM / |n ∪ m|
        const float Dthr = std::min(Da, Db) + KD / static_cast<float>(s);
        const float Mthr = std::min(Ma, Mb) + KM / static_cast<float>(s);

        return (Dunion <= Dthr) && (Munion <= Mthr);
    }

    void merge(int ra, int rb) {
        if (size[ra] < size[rb])
            std::swap(ra, rb);

        const float ca = center(min_ang[ra], max_ang[ra]);
        const float cb = center(min_ang[rb], max_ang[rb]);
        const float sh = compute_shift_360(ca, cb);

        const float minB = min_ang[rb] + sh;
        const float maxB = max_ang[rb] + sh;

        parent[rb] = ra;
        size[ra] += size[rb];

        min_mag[ra] = std::min(min_mag[ra], min_mag[rb]);
        max_mag[ra] = std::max(max_mag[ra], max_mag[rb]);

        min_ang[ra] = std::min(min_ang[ra], minB);
        max_ang[ra] = std::max(max_ang[ra], maxB);
    }
};

struct Edge {
    int a = 0;
    int b = 0;
    float w = 0.0f;
};

static UnionFind cluster_gradients_olson2011_8conn(const Gradients &g, float KD, float KM) {
    const int w = g.mag.width();
    const int h = g.mag.height();
    rassert(g.mag.channels() == 1 && g.angle.channels() == 1, "cluster_gradients expects 1-channel mag/angle");
    rassert(g.angle.width() == w && g.angle.height() == h, "mag/angle size mismatch");

    UnionFind uf(w, h);

    // Initialize per-pixel component stats.
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int idx = j * w + i;
            const float a = wrap_angle_deg(g.angle(j, i));
            const float m = g.mag(j, i);

            uf.min_ang[idx] = a;
            uf.max_ang[idx] = a;

            uf.min_mag[idx] = m;
            uf.max_mag[idx] = m;
        }
    }

    // 8-connected adjacency, stored once per undirected edge:
    // right, down, down-right, down-left.
    const std::size_t n_edges = static_cast<std::size_t>((w - 1) * h) + static_cast<std::size_t>(w * (h - 1)) +
                                static_cast<std::size_t>(2 * (w - 1) * (h - 1));

    std::vector<Edge> edges;
    edges.reserve(n_edges);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int a = j * w + i;

            if (i + 1 < w) {
                const int b = j * w + (i + 1);
                edges.push_back(Edge{a, b, angle_diff_deg(g.angle(j, i), g.angle(j, i + 1))});
            }
            if (j + 1 < h) {
                const int b = (j + 1) * w + i;
                edges.push_back(Edge{a, b, angle_diff_deg(g.angle(j, i), g.angle(j + 1, i))});
            }
            if (i + 1 < w && j + 1 < h) {
                const int b = (j + 1) * w + (i + 1);
                edges.push_back(Edge{a, b, angle_diff_deg(g.angle(j, i), g.angle(j + 1, i + 1))});
            }
            if (i - 1 >= 0 && j + 1 < h) {
                const int b = (j + 1) * w + (i - 1);
                edges.push_back(Edge{a, b, angle_diff_deg(g.angle(j, i), g.angle(j + 1, i - 1))});
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge &e1, const Edge &e2) { return e1.w < e2.w; });

    for (const Edge &e : edges) {
        int ra = uf.find(e.a);
        int rb = uf.find(e.b);
        if (ra == rb)
            continue;

        if (uf.can_merge(ra, rb, KD, KM)) {
            uf.merge(ra, rb);
        }
    }

    return uf;
}

struct Segment {
    float x0 = 0.0f, y0 = 0.0f;
    float x1 = 0.0f, y1 = 0.0f;
    float dir_x = 1.0f, dir_y = 0.0f; // Unit direction along the segment (from p0 to p1).
};

static std::vector<Segment> fit_segments_weighted_ls(const Gradients &g, const UnionFind &uf, int min_pixels,
                                                     float min_mean_mag, float min_length) {
    const int w = uf.w;
    const int h = uf.h;

    std::unordered_map<int, std::vector<int>> pixels;
    pixels.reserve(static_cast<std::size_t>(w * h / 16));

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int idx = j * w + i;
            const int r = uf.find(idx);
            pixels[r].push_back(idx);
        }
    }

    std::vector<Segment> segments;
    segments.reserve(pixels.size());

    for (auto &kv : pixels) {
        const std::vector<int> &pts = kv.second;
        if (static_cast<int>(pts.size()) < min_pixels)
            continue;

        // Weighted sums.
        double sw = 0.0;
        double sx = 0.0, sy = 0.0;
        double sxx = 0.0, sxy = 0.0, syy = 0.0;
        double sdx = 0.0, sdy = 0.0;

        for (int idx : pts) {
            const int y = idx / w;
            const int x = idx - y * w;

            const float wgt = std::max(g.mag(y, x), 0.0f);
            sw += wgt;
            sx += wgt * static_cast<double>(x);
            sy += wgt * static_cast<double>(y);

            sxx += wgt * static_cast<double>(x) * static_cast<double>(x);
            sxy += wgt * static_cast<double>(x) * static_cast<double>(y);
            syy += wgt * static_cast<double>(y) * static_cast<double>(y);

            sdx += wgt * static_cast<double>(g.dx(y, x));
            sdy += wgt * static_cast<double>(g.dy(y, x));
        }

        if (sw <= 1e-9)
            continue;

        const double mean_mag = sw / static_cast<double>(pts.size());
        if (mean_mag < static_cast<double>(min_mean_mag))
            continue;

        const double cx = sx / sw;
        const double cy = sy / sw;

        const double exx = sxx / sw - cx * cx;
        const double exy = sxy / sw - cx * cy;
        const double eyy = syy / sw - cy * cy;

        // Principal direction via analytic eigenvector for 2x2 covariance.
        const double theta = 0.5 * std::atan2(2.0 * exy, exx - eyy);
        double dx_line = std::cos(theta);
        double dy_line = std::sin(theta);

        const double norm = std::sqrt(dx_line * dx_line + dy_line * dy_line);
        if (norm <= 1e-12)
            continue;
        dx_line /= norm;
        dy_line /= norm;

        // Segment endpoints via projection min/max.
        double tmin = std::numeric_limits<double>::infinity();
        double tmax = -std::numeric_limits<double>::infinity();
        for (int idx : pts) {
            const int y = idx / w;
            const int x = idx - y * w;
            const double px = static_cast<double>(x) - cx;
            const double py = static_cast<double>(y) - cy;
            const double t = px * dx_line + py * dy_line;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }

        const double x0 = cx + tmin * dx_line;
        const double y0 = cy + tmin * dy_line;
        const double x1 = cx + tmax * dx_line;
        const double y1 = cy + tmax * dy_line;

        const double len = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        if (len < static_cast<double>(min_length))
            continue;

        // Orient segment so that dark is on the left and light is on the right.
        // Gradient points from dark to light. For an oriented segment direction d,
        // left normal is n_left = (-dy, dx). We want n_left · g < 0 (g points to the right).
        const double gavgx = sdx / sw;
        const double gavgy = sdy / sw;

        const double n_left_x = -dy_line;
        const double n_left_y = dx_line;
        const double dot = n_left_x * gavgx + n_left_y * gavgy;

        Segment seg;
        if (dot > 0.0) {
            // Flip orientation.
            seg.x0 = static_cast<float>(x1);
            seg.y0 = static_cast<float>(y1);
            seg.x1 = static_cast<float>(x0);
            seg.y1 = static_cast<float>(y0);
            seg.dir_x = static_cast<float>(-dx_line);
            seg.dir_y = static_cast<float>(-dy_line);
        } else {
            seg.x0 = static_cast<float>(x0);
            seg.y0 = static_cast<float>(y0);
            seg.x1 = static_cast<float>(x1);
            seg.y1 = static_cast<float>(y1);
            seg.dir_x = static_cast<float>(dx_line);
            seg.dir_y = static_cast<float>(dy_line);
        }

        segments.push_back(seg);
    }

    return segments;
}

static std::uint8_t hash_u8(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<std::uint8_t>(x & 0xFFU);
}

static image8u visualize_components(const UnionFind &uf) {
    const int w = uf.w;
    const int h = uf.h;

    image8u out(w, h, 3);

    std::unordered_map<int, int> root_to_id;
    root_to_id.reserve(static_cast<std::size_t>(w * h / 8));
    int next_id = 0;

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int idx = j * w + i;
            const int r = uf.find(idx);
            auto it = root_to_id.find(r);
            if (it == root_to_id.end())
                it = root_to_id.emplace(r, next_id++).first;

            const std::uint32_t id = static_cast<std::uint32_t>(it->second);
            out(j, i, 0) = hash_u8(id * 3U + 0U);
            out(j, i, 1) = hash_u8(id * 3U + 1U);
            out(j, i, 2) = hash_u8(id * 3U + 2U);
        }
    }

    return out;
}

static image8u to_rgb_for_overlay(const image8u &img) {
    const int w = img.width();
    const int h = img.height();

    if (img.channels() == 3)
        return img;

    image8u out(w, h, 3);
    if (img.channels() == 1) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                const std::uint8_t v = img(j, i);
                out(j, i, 0) = v;
                out(j, i, 1) = v;
                out(j, i, 2) = v;
            }
        }
        return out;
    }

    rassert(img.channels() == 4, "Unexpected channels for overlay", img.channels());
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            out(j, i, 0) = img(j, i, 0);
            out(j, i, 1) = img(j, i, 1);
            out(j, i, 2) = img(j, i, 2);
        }
    }
    return out;
}

static void draw_line(image8u &rgb, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    rassert(rgb.channels() == 3, "draw_line expects RGB", rgb.channels());

    const int w = rgb.width();
    const int h = rgb.height();

    auto set_px = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h)
            return;
        rgb(y, x, 0) = r;
        rgb(y, x, 1) = g;
        rgb(y, x, 2) = b;
    };

    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    int x = x0;
    int y = y0;
    while (true) {
        set_px(x, y);
        if (x == x1 && y == y1)
            break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

static image8u visualize_segments_overlay(const image8u &input_rgb, const std::vector<Segment> &segs,
                                          bool draw_notches) {
    image8u out = to_rgb_for_overlay(input_rgb);

    for (const Segment &s : segs) {
        const float dx = s.x1 - s.x0;
        const float dy = s.y1 - s.y0;
        const float a = wrap_angle_deg(std::atan2(dy, dx) * 180.0f / kPi);
        std::uint8_t rr, gg, bb;
        hsv_to_rgb(a / 360.0f, 1.0f, 1.0f, &rr, &gg, &bb);

        const int x0 = static_cast<int>(std::lround(s.x0));
        const int y0 = static_cast<int>(std::lround(s.y0));
        const int x1 = static_cast<int>(std::lround(s.x1));
        const int y1 = static_cast<int>(std::lround(s.y1));

        draw_line(out, x0, y0, x1, y1, rr, gg, bb);

        if (draw_notches) {
            // Notch points towards the lighter region.
            // With dark-left/light-right convention, the right normal points to light.
            const float ndx = s.dir_y;  // right normal x
            const float ndy = -s.dir_x; // right normal y

            const float mx = 0.5f * (s.x0 + s.x1);
            const float my = 0.5f * (s.y0 + s.y1);

            const float notch_len = cfg::kNotchLengthPx;
            const int nx0 = static_cast<int>(std::lround(mx));
            const int ny0 = static_cast<int>(std::lround(my));
            const int nx1 = static_cast<int>(std::lround(mx + notch_len * ndx));
            const int ny1 = static_cast<int>(std::lround(my + notch_len * ndy));

            draw_line(out, nx0, ny0, nx1, ny1, rr, gg, bb);
        }
    }

    return out;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  sobel_app <input_image.(png|jpg|jpeg)> <output_dir|output_file>\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        fs::path out_path = argv[2];

        fs::path out_dir = out_path;
        if (out_path.has_extension()) {
            out_dir = out_path.parent_path();
            if (out_dir.empty())
                out_dir = ".";
        }
        fs::create_directories(out_dir);

        // 00: input
        image8u input = libimages::load_image(input_path.string());

        // downscaled input (working image)
        input = downscale_nearest(input, cfg::kDownscaleRatio);
        libimages::debug_io::dump_image(
            stage_path(out_dir, 1, "input_downscaled_r_" + std::to_string(cfg::kDownscaleRatio)), input,
            cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

        libimages::debug_io::dump_image(stage_path(out_dir, 0, "input_image"), input, cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);

        // 01: grayscale
        const image32f gray = to_grayscale_float(input);
        libimages::debug_io::dump_image(stage_path(out_dir, 1, "gray"), gray, cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);

        // 02: blurred grayscale
        const image32f blurred = gaussian_blur_gray(gray, cfg::kGaussianSigma);
        libimages::debug_io::dump_image(
            stage_path(out_dir, 2, "gray_blurred_sigma_" + std::to_string(cfg::kGaussianSigma)), blurred,
            cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

        // 03..: gradients
        const Gradients gr = sobel_gradients(blurred);

        libimages::debug_io::dump_image(stage_path(out_dir, 3, "sobel_dx_signed"), visualize_signed_to_u8(gr.dx),
                                        cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
        libimages::debug_io::dump_image(stage_path(out_dir, 4, "sobel_dy_signed"), visualize_signed_to_u8(gr.dy),
                                        cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
        libimages::debug_io::dump_image(stage_path(out_dir, 5, "sobel_mag"), gr.mag, cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);
        libimages::debug_io::dump_image(stage_path(out_dir, 6, "sobel_angle_hsv"),
                                        visualize_angle_hsv(gr.angle, gr.mag), cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);

        // 07: clustering into components (8-connected)
        const UnionFind uf = cluster_gradients_olson2011_8conn(gr, cfg::kClusterKD, cfg::kClusterKM);
        libimages::debug_io::dump_image(stage_path(out_dir, 7, "gradient_components_random_colors"),
                                        visualize_components(uf), cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

        // 08..: fit directed line segments and visualize
        const std::vector<Segment> segs = fit_segments_weighted_ls(gr, uf, cfg::kMinComponentPixels,
                                                                   cfg::kMinMeanMagnitude, cfg::kMinSegmentLengthPx);

        libimages::debug_io::dump_image(stage_path(out_dir, 8, "segments_overlay"),
                                        visualize_segments_overlay(input, segs, false), cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);

        libimages::debug_io::dump_image(stage_path(out_dir, 9, "segments_overlay_with_notches"),
                                        visualize_segments_overlay(input, segs, true), cfg::kDumpVerbose,
                                        cfg::kDumpForceOverwrite);

        std::cout << "Saved debug images to: " << out_dir.string() << "\n";
        std::cout << "Segments: " << segs.size() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
