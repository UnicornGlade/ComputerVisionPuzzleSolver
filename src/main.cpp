#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

#include <libimages/algorithms/downscale.h>
#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include "algorithms/background_masking.h"
#include "algorithms/utils.h"
#include "algorithms/find_mask_borders.h"

namespace fs = std::filesystem;

namespace cfg {

// 1) Load + downscale
inline constexpr float kDownscaleRatio = 4.0f;

// 2) Blur
inline constexpr float kGaussianSigma = 0.5f;

// 3) Mask refinement
inline constexpr int kMaskDilateStrength = std::max((int) 2, (int) (32.0f / kDownscaleRatio));
inline constexpr int kMaskErodeStrength = kMaskDilateStrength + 2;

// 4) Extract objects
inline constexpr bool kObjectsEightConnected = true;

// 5) Border/sides/sampling
inline constexpr int kBorderMinBgNeighbors = 2; // >=2 background neighbors => border pixel
inline constexpr int kBorderSideCount = 4;      // K
inline constexpr int kSamplesPerSide = 40;      // L

// 6) Side matching
inline constexpr int kTopNeighborsPerSide = 10;
inline constexpr float kSideColorBlurSigma = 1.0f; // 0 => disable
inline constexpr bool kCompareReversedOnly = true; // as you requested

// 6b) Monochrome side detection (skip matching/visualization for such sides)
inline constexpr float kMonoMaxChannelRange = 35.0f; // if max(R,G,B) range <= this -> monochrome
inline constexpr float kMonoQuantile = 0.99f;        // robust: look at 90% quantile of RGB distance-to-mean
inline constexpr float kMonoQuantileMaxDist = 10.0f; // if q90 dist <= this -> monochrome
inline constexpr int   kMonoMinLen = 6;              // short sides treated as monochrome

// High-level outputs
inline constexpr const char* kOut00 = "00_input_downscaled.png";
inline constexpr const char* kOut01 = "01_blur_gray.png";
inline constexpr const char* kOut02 = "02_foreground_mask.png";
inline constexpr const char* kOut03 = "03_objects_overlay.png";
inline constexpr const char* kOut04 = "04_best_side_matches.png";

// Low-level folders
inline constexpr const char* kDbg01 = "01_load_downscale";
inline constexpr const char* kDbg02 = "02_blur";
inline constexpr const char* kDbg03 = "03_background_mask";
inline constexpr const char* kDbg04 = "04_extract_objects";
inline constexpr const char* kDbg05 = "05_find_mask_borders";

inline constexpr const char* kLL00 = "00_input_downscaled.png";
inline constexpr const char* kLL10 = "10_gray_u8.png";
inline constexpr const char* kLL20 = "20_blur_gray_u8.png";
inline constexpr const char* kLL30 = "30_mask_raw.png";
inline constexpr const char* kLL31 = "31_mask_refined.png";
inline constexpr const char* kLL40 = "40_objects_overlay.png";

inline constexpr std::uint32_t kVizSeed = 123;

} // namespace cfg

using libimages::image8u;
using libimages::image32f;

static std::string idx4(std::size_t k) {
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << k;
    return ss.str();
}

static image8u gray32f_to_u8_clamp(const image32f& gray) {
    rassert(gray.channels() == 1, "gray32f_to_u8_clamp expects 1-channel image32f", gray.channels());

    image8u out(gray.width(), gray.height(), 1);
    for (int j = 0; j < gray.height(); ++j) {
        for (int i = 0; i < gray.width(); ++i) {
            float v = gray(j, i);
            if (std::isnan(v) || std::isinf(v)) v = 0.0f;
            v = std::clamp(v, 0.0f, 255.0f);
            out(j, i) = static_cast<std::uint8_t>(std::lround(v));
        }
    }
    return out;
}

static image8u make_raw_threshold_mask(const image8u& gray_u8, float thr) {
    rassert(gray_u8.channels() == 1, "make_raw_threshold_mask expects 1-channel image8u", gray_u8.channels());
    image8u m(gray_u8.width(), gray_u8.height(), 1);
    for (int j = 0; j < gray_u8.height(); ++j) {
        for (int i = 0; i < gray_u8.width(); ++i) {
            m(j, i) = (static_cast<float>(gray_u8(j, i)) > thr) ? 255 : 0;
        }
    }
    return m;
}

static image8u make_dark_rgb(const image8u& input) {
    const int w = input.width();
    const int h = input.height();
    rassert(input.channels() == 1 || input.channels() == 3 || input.channels() == 4, "Unsupported input channels",
            input.channels());

    image8u out(w, h, 3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            std::uint8_t r = 0, g = 0, b = 0;
            if (input.channels() == 1) {
                const std::uint8_t v = input(j, i);
                r = g = b = static_cast<std::uint8_t>(v / 2);
            } else {
                r = static_cast<std::uint8_t>(input(j, i, 0) / 2);
                g = static_cast<std::uint8_t>(input(j, i, 1) / 2);
                b = static_cast<std::uint8_t>(input(j, i, 2) / 2);
            }
            out(j, i, 0) = r;
            out(j, i, 1) = g;
            out(j, i, 2) = b;
        }
    }
    return out;
}

static image8u mask_to_rgb_dim(const image8u& mask, std::uint8_t v_obj = 120) {
    rassert(mask.channels() == 1, "mask_to_rgb_dim expects 1-channel mask", mask.channels());
    image8u out(mask.width(), mask.height(), 3);
    out.fill(std::uint8_t(0));
    for (int j = 0; j < mask.height(); ++j) {
        for (int i = 0; i < mask.width(); ++i) {
            if (mask(j, i) == 255) {
                out(j, i, 0) = v_obj;
                out(j, i, 1) = v_obj;
                out(j, i, 2) = v_obj;
            }
        }
    }
    return out;
}

// -------------------- simple drawing (for global high-level) --------------------

static inline bool in_bounds(int x, int y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

static void put_pixel_rgb_thick(image8u& img, int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                int thickness) {
    const int w = img.width();
    const int h = img.height();
    if (img.channels() != 3) return;

    const int rad = std::max(0, thickness - 1);
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            const int xx = x + dx;
            const int yy = y + dy;
            if (!in_bounds(xx, yy, w, h)) continue;
            img(yy, xx, 0) = r;
            img(yy, xx, 1) = g;
            img(yy, xx, 2) = b;
        }
    }
}

static void draw_line_rgb(image8u& img, point2i a, point2i b,
                          std::uint8_t r, std::uint8_t g, std::uint8_t bl,
                          int thickness) {
    int x0 = a.x, y0 = a.y;
    const int x1 = b.x, y1 = b.y;

    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        put_pixel_rgb_thick(img, x0, y0, r, g, bl, thickness);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// -------------------- side colors matching --------------------

struct Color3f final { float r = 0, g = 0, b = 0; };

static Color3f to_f(const find_mask_borders::point3u& c) {
    return Color3f{float(c.r), float(c.g), float(c.b)}; // point3u fields assumed x,y,z (as in your earlier style)
}

static find_mask_borders::point3u to_u8(const Color3f& c) {
    auto cl = [](float v) -> std::uint8_t {
        v = std::clamp(v, 0.0f, 255.0f);
        return static_cast<std::uint8_t>(std::lround(v));
    };
    return find_mask_borders::point3u{cl(c.r), cl(c.g), cl(c.b)};
}

static std::vector<float> gaussian_kernel_1d(float sigma) {
    if (!(sigma > 0.0f)) return {1.0f};
    const int rad = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> k(static_cast<std::size_t>(2 * rad + 1), 0.0f);
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;
    for (int i = -rad; i <= rad; ++i) {
        const float w = std::exp(-float(i * i) * inv2s2);
        k[static_cast<std::size_t>(i + rad)] = w;
        sum += w;
    }
    for (auto& v : k) v /= sum;
    return k;
}

static std::vector<Color3f> blur_colors_1d(const std::vector<find_mask_borders::point3u>& v, float sigma) {
    const int n = static_cast<int>(v.size());
    std::vector<Color3f> out;
    out.resize(v.size());

    if (n == 0) return out;
    if (!(sigma > 0.0f) || n == 1) {
        for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = to_f(v[static_cast<std::size_t>(i)]);
        return out;
    }

    const auto k = gaussian_kernel_1d(sigma);
    const int rad = (static_cast<int>(k.size()) - 1) / 2;

    for (int i = 0; i < n; ++i) {
        float sr = 0, sg = 0, sb = 0;
        for (int t = -rad; t <= rad; ++t) {
            int j = i + t;
            if (j < 0) j = 0;
            if (j >= n) j = n - 1;
            const float w = k[static_cast<std::size_t>(t + rad)];
            const auto cf = to_f(v[static_cast<std::size_t>(j)]);
            sr += w * cf.r;
            sg += w * cf.g;
            sb += w * cf.b;
        }
        out[static_cast<std::size_t>(i)] = Color3f{sr, sg, sb};
    }
    return out;
}

static bool is_almost_monochrome(const std::vector<Color3f>& v,
                                 float max_channel_range,
                                 float quantile,
                                 float max_q_dist,
                                 int min_len) {
    const int n = static_cast<int>(v.size());
    if (n <= 0) return true;
    if (n < min_len) return true;

    // Fast range check (robust for "nearly constant" sides)
    float minr = v[0].r, maxr = v[0].r;
    float ming = v[0].g, maxg = v[0].g;
    float minb = v[0].b, maxb = v[0].b;

    double mr = 0.0, mg = 0.0, mb = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& c = v[static_cast<std::size_t>(i)];
        minr = std::min(minr, c.r); maxr = std::max(maxr, c.r);
        ming = std::min(ming, c.g); maxg = std::max(maxg, c.g);
        minb = std::min(minb, c.b); maxb = std::max(maxb, c.b);
        mr += c.r; mg += c.g; mb += c.b;
    }

    const float rr = maxr - minr;
    const float gr = maxg - ming;
    const float br = maxb - minb;
    const float rmax = std::max(rr, std::max(gr, br));
    if (rmax <= max_channel_range) return true;

    // Robust quantile of RGB distance to mean
    mr /= double(n); mg /= double(n); mb /= double(n);

    std::vector<float> d2;
    d2.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& c = v[static_cast<std::size_t>(i)];
        const float dr = float(double(c.r) - mr);
        const float dg = float(double(c.g) - mg);
        const float db = float(double(c.b) - mb);
        d2.push_back(dr * dr + dg * dg + db * db);
    }

    float q = std::clamp(quantile, 0.0f, 1.0f);
    const std::size_t qi = static_cast<std::size_t>(std::lround(q * float(n - 1)));
    std::nth_element(d2.begin(), d2.begin() + static_cast<std::ptrdiff_t>(qi), d2.end());

    const float qdist = std::sqrt(std::max(0.0f, d2[qi]));
    return (qdist <= max_q_dist);
}


// ZNCC distance between two color sequences of equal length, comparing a[i] vs b[L-1-i] (reverse b).
// Returns dist = 1 - corr_avg, where corr_avg in [-1,1]. Smaller is better.
static float distance_colors_reversed_zncc(const std::vector<Color3f>& a, const std::vector<Color3f>& b) {
    rassert(a.size() == b.size(), "distance_colors_reversed_zncc: size mismatch", a.size(), b.size());
    const int L = static_cast<int>(a.size());
    if (L == 0) return 1.0f;
    if (L == 1) {
        const auto& ca = a[0];
        const auto& cb = b[0];
        const float dr = ca.r - cb.r;
        const float dg = ca.g - cb.g;
        const float db = ca.b - cb.b;
        const float d = std::sqrt(dr * dr + dg * dg + db * db);
        // Map "equal -> 0", "different -> ~1"
        return (d < 1e-6f) ? 0.0f : 1.0f;
    }

    auto zncc_channel = [&](auto get) -> float {
        double meanA = 0.0, meanB = 0.0;
        for (int i = 0; i < L; ++i) {
            meanA += static_cast<double>(get(a[static_cast<std::size_t>(i)]));
            meanB += static_cast<double>(get(b[static_cast<std::size_t>(L - 1 - i)]));
        }
        meanA /= static_cast<double>(L);
        meanB /= static_cast<double>(L);

        double num = 0.0;
        double denA = 0.0;
        double denB = 0.0;
        for (int i = 0; i < L; ++i) {
            const double xa = static_cast<double>(get(a[static_cast<std::size_t>(i)])) - meanA;
            const double xb = static_cast<double>(get(b[static_cast<std::size_t>(L - 1 - i)])) - meanB;
            num += xa * xb;
            denA += xa * xa;
            denB += xb * xb;
        }

        const double eps = 1e-12;
        const double denom = std::sqrt(std::max(denA * denB, eps));

        // If variance is ~0, correlation is ill-defined. Handle robustly:
        // - if both are (almost) constant and means close -> corr=1
        // - else corr=0
        if (denA < eps || denB < eps) {
            const double dm = std::abs(meanA - meanB);
            return (dm < 1e-6) ? 1.0f : 0.0f;
        }

        double corr = num / denom;
        corr = std::clamp(corr, -1.0, 1.0);
        return static_cast<float>(corr);
    };

    const float cr = zncc_channel([](const Color3f& c) { return c.r; });
    const float cg = zncc_channel([](const Color3f& c) { return c.g; });
    const float cb = zncc_channel([](const Color3f& c) { return c.b; });

    const float corr_avg = (cr + cg + cb) / 3.0f;
    // Distance: 0 best, 2 worst
    return 1.0f - corr_avg;
}

static point2i centroid_round(const std::vector<point2i>& pix) {
    if (pix.empty()) return point2i{0, 0};
    double sx = 0.0, sy = 0.0;
    for (const auto& p : pix) { sx += p.x; sy += p.y; }
    const double cx = sx / double(pix.size());
    const double cy = sy / double(pix.size());
    return point2i{static_cast<int>(std::lround(cx)), static_cast<int>(std::lround(cy))};
}

static image8u side_overlay_single(const image8u& base_rgb, const std::vector<point2i>& side, std::uint32_t seed) {
    std::vector<std::vector<point2i>> one;
    one.push_back(side);
    return find_mask_borders::visualize_sides_overlay(base_rgb, one, seed, 2);
}

static image8u concat_h(const image8u& a, const image8u& b) {
    rassert(a.channels() == 3 && b.channels() == 3, "concat_h expects RGB images");
    const int w = a.width() + b.width();
    const int h = std::max(a.height(), b.height());
    image8u out(w, h, 3);
    out.fill(std::uint8_t(0));

    auto paste = [&](const image8u& src, int x0) {
        for (int y = 0; y < src.height(); ++y) {
            for (int x = 0; x < src.width(); ++x) {
                out(y, x0 + x, 0) = src(y, x, 0);
                out(y, x0 + x, 1) = src(y, x, 1);
                out(y, x0 + x, 2) = src(y, x, 2);
            }
        }
    };

    paste(a, 0);
    paste(b, a.width());
    return out;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  app <input.(png|jpg|jpeg)> <output_dir> [debug_root]\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        const fs::path debug_root = (argc >= 4) ? fs::path(argv[3]) : (out_dir / "debug-low");
        fs::create_directories(debug_root);

        const fs::path dbg1 = debug_root / cfg::kDbg01;
        const fs::path dbg2 = debug_root / cfg::kDbg02;
        const fs::path dbg3 = debug_root / cfg::kDbg03;
        const fs::path dbg4 = debug_root / cfg::kDbg04;
        const fs::path dbg5 = debug_root / cfg::kDbg05;
        fs::create_directories(dbg1);
        fs::create_directories(dbg2);
        fs::create_directories(dbg3);
        fs::create_directories(dbg4);
        fs::create_directories(dbg5);

        // =========================================================
        // 1) Read input image and downscale
        // =========================================================
        image8u input = libimages::load_image(input_path.string());
        input = libimages::downscale_area(input, cfg::kDownscaleRatio);

        libimages::debug_io::dump_image((out_dir / cfg::kOut00).string(), input, true, true);
        libimages::debug_io::dump_image((dbg1 / cfg::kLL00).string(), input, true, true);

        // =========================================================
        // 2) Blur (on grayscale)
        // =========================================================
        const image32f gray_f = libimages::to_grayscale_float(input);
        const image8u gray_u8 = gray32f_to_u8_clamp(gray_f);
        libimages::debug_io::dump_image((dbg2 / cfg::kLL10).string(), gray_u8, true, true);

        const image32f blur_f = libimages::gaussian_blur_gray(gray_f, cfg::kGaussianSigma);
        const image8u blur_u8 = gray32f_to_u8_clamp(blur_f);

        libimages::debug_io::dump_image((out_dir / cfg::kOut01).string(), blur_u8, true, true);
        libimages::debug_io::dump_image((dbg2 / cfg::kLL20).string(), blur_u8, true, true);

        // =========================================================
        // 3) Build foreground mask using background masking
        // =========================================================
        background_masking::Params mp;
        mp.dilate_strength = cfg::kMaskDilateStrength;
        mp.erode_strength = cfg::kMaskErodeStrength;

        const float thr = background_masking::estimate_background_threshold(blur_u8);
        const image8u mask_raw = make_raw_threshold_mask(blur_u8, thr);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL30).string(), mask_raw, true, true);

        const image8u mask = background_masking::build_foreground_mask(blur_u8, mp);

        libimages::debug_io::dump_image((out_dir / cfg::kOut02).string(), mask, true, true);
        libimages::debug_io::dump_image((dbg3 / cfg::kLL31).string(), mask, true, true);

        // =========================================================
        // 4) Split into connected foreground objects + save each as image+mask (flat)
        // =========================================================
        utils::ExtractParams ep;
        ep.eight_connected = cfg::kObjectsEightConnected;

        const auto objects = utils::extract_objects_by_mask<std::uint8_t>(input, mask, ep);

        const fs::path out_objs = out_dir / "objects";
        const fs::path dbg_objs = dbg4 / "objects";
        fs::create_directories(out_objs);
        fs::create_directories(dbg_objs);

        image8u hi_overlay = make_dark_rgb(input);
        FastRandom rng(cfg::kVizSeed);

        for (std::size_t k = 0; k < objects.size(); ++k) {
            const auto& obj = objects[k];
            const std::string pre = idx4(k);

            libimages::debug_io::dump_image((out_objs / (pre + "image.png")).string(), obj.image, true, true);
            libimages::debug_io::dump_image((out_objs / (pre + "mask.png")).string(), obj.mask, true, true);

            libimages::debug_io::dump_image((dbg_objs / (pre + "image.png")).string(), obj.image, true, true);
            libimages::debug_io::dump_image((dbg_objs / (pre + "mask.png")).string(), obj.mask, true, true);

            const std::uint32_t rv = rng.nextU32();
            const std::uint8_t cr = static_cast<std::uint8_t>(rv & 0xFFu);
            const std::uint8_t cg = static_cast<std::uint8_t>((rv >> 8) & 0xFFu);
            const std::uint8_t cb = static_cast<std::uint8_t>((rv >> 16) & 0xFFu);

            for (int jj = 0; jj < obj.mask.height(); ++jj) {
                for (int ii = 0; ii < obj.mask.width(); ++ii) {
                    if (obj.mask(jj, ii) != 255) continue;
                    const int y = obj.offset.y + jj;
                    const int x = obj.offset.x + ii;
                    if (!in_bounds(x, y, hi_overlay.width(), hi_overlay.height())) continue;
                    hi_overlay(y, x, 0) = cr;
                    hi_overlay(y, x, 1) = cg;
                    hi_overlay(y, x, 2) = cb;
                }
            }
        }

        libimages::debug_io::dump_image((out_dir / cfg::kOut03).string(), hi_overlay, true, true);
        libimages::debug_io::dump_image((dbg4 / cfg::kLL40).string(), hi_overlay, true, true);

        // =========================================================
        // 5) For each object -> border mask -> K sides -> sample KxL colors
        //    AND store all per-object data for global matching
        // =========================================================
        struct ObjSides final {
            std::vector<std::vector<point2i>> sides_pix;                    // K
            std::vector<std::vector<find_mask_borders::point3u>> samples;   // KxL
            std::vector<std::vector<Color3f>> samples_blur;                 // KxL blurred
            std::vector<char> side_is_mono;                                 // K (0/1)
            std::vector<point2i> side_centers_local;                        // K
            std::vector<point2i> side_centers_global;                       // K
        };

        std::vector<ObjSides> all;
        all.resize(objects.size());

        const fs::path dbg_sides = dbg5 / "objects_sides";
        fs::create_directories(dbg_sides);

        for (std::size_t k = 0; k < objects.size(); ++k) {
            const auto& obj = objects[k];
            const std::string pre = idx4(k);

            const image8u border = find_mask_borders::border_pixels_mask(obj.mask, cfg::kBorderMinBgNeighbors);
            libimages::debug_io::dump_image((dbg_sides / (pre + "border.png")).string(), border, true, true);

            find_mask_borders::SplitSidesDebugParams sd;
            sd.out_dir = dbg_sides;
            sd.prefix = pre + "_";
            const auto sides = find_mask_borders::split_border_into_sides(obj.mask, border, cfg::kBorderSideCount, &sd);

            find_mask_borders::SamplingDebugParams dp;
            dp.out_dir = dbg_sides;
            dp.prefix = pre + "_";
            dp.dump_ext = ".png";
            dp.force_overwrite = true;
            dp.verbose = false;
            dp.point_radius = 1;

            const auto samples = find_mask_borders::sample_sides_colors(obj.image, sides, cfg::kSamplesPerSide, &dp);

            all[k].sides_pix = sides;
            all[k].samples = samples;

            all[k].side_centers_local.clear();
            all[k].side_centers_global.clear();
            all[k].side_centers_local.reserve(all[k].sides_pix.size());
            all[k].side_centers_global.reserve(all[k].sides_pix.size());

            for (const auto& s : all[k].sides_pix) {
                const point2i c = centroid_round(s);
                all[k].side_centers_local.push_back(c);
                all[k].side_centers_global.push_back(point2i{obj.offset.x + c.x, obj.offset.y + c.y});
            }

            // Blur color sequences (per side)
            all[k].samples_blur.clear();
            all[k].side_is_mono.clear();
            all[k].samples_blur.reserve(all[k].samples.size());
            all[k].side_is_mono.reserve(all[k].samples.size());

            for (const auto& row : all[k].samples) {
                auto blurred = blur_colors_1d(row, cfg::kSideColorBlurSigma);
                const bool mono = is_almost_monochrome(blurred,
                                                       cfg::kMonoMaxChannelRange,
                                                       cfg::kMonoQuantile,
                                                       cfg::kMonoQuantileMaxDist,
                                                       cfg::kMonoMinLen);
                all[k].samples_blur.push_back(std::move(blurred));
                all[k].side_is_mono.push_back(static_cast<char>(mono ? 1 : 0));
            }
        }

        // =========================================================
        // 6) NEW: for each object A and each side B -> find top-10 similar sides of other objects
        //    Save per-object folders + per-side subfolders with visualizations
        // =========================================================
        const fs::path out_match_root = out_dir / "objects_side_matches";
        fs::create_directories(out_match_root);

        // high-level: draw lines from each side center to best match side center
        image8u hi_best = input;

        for (std::size_t a = 0; a < objects.size(); ++a) {
            const std::string A = idx4(a);
            const fs::path dirA = out_match_root / A;
            fs::create_directories(dirA);

            // Save object image/mask + sides-on-mask visualization
            libimages::debug_io::dump_image((dirA / "image.png").string(), objects[a].image, true, true);
            libimages::debug_io::dump_image((dirA / "mask.png").string(), objects[a].mask, true, true);

            {
                const image8u base = mask_to_rgb_dim(objects[a].mask, 120);
                const image8u sides_on_mask =
                    find_mask_borders::visualize_sides_overlay(base, all[a].sides_pix, cfg::kVizSeed ^ std::uint32_t(a), 2);
                libimages::debug_io::dump_image((dirA / "sides_on_mask.png").string(), sides_on_mask, true, true);
            }

            const int Ka = static_cast<int>(all[a].samples_blur.size());
            for (int b = 0; b < Ka; ++b) {
                const bool monoA = (b >= 0 && b < static_cast<int>(all[a].side_is_mono.size())) ? (all[a].side_is_mono[static_cast<std::size_t>(b)] != 0) : false;
                if (monoA) {
                    // Side is nearly monochrome -> skip neighbors visualization + skip high-level arrow
                    continue;
                }

                const fs::path dirB = dirA / ("side" + std::to_string(b));
                fs::create_directories(dirB);

                struct Cand { std::size_t c; int d; float dist; };
                std::vector<Cand> cands;
                cands.reserve(256);

                for (std::size_t c = 0; c < objects.size(); ++c) {
                    if (c == a) continue;
                    const int Kc = static_cast<int>(all[c].samples_blur.size());
                    for (int d = 0; d < Kc; ++d) {
                        const bool monoC = (d >= 0 && d < static_cast<int>(all[c].side_is_mono.size())) ? (all[c].side_is_mono[static_cast<std::size_t>(d)] != 0) : false;
                        if (monoC) continue;

                        if (all[a].samples_blur[static_cast<std::size_t>(b)].size() != all[c].samples_blur[static_cast<std::size_t>(d)].size())
                            continue;

                        float dist_rev = distance_colors_reversed_zncc(all[a].samples_blur[static_cast<std::size_t>(b)],
                                                                       all[c].samples_blur[static_cast<std::size_t>(d)]);
                        float dist = dist_rev;
                        cands.push_back(Cand{c, d, dist});
                    }
                }

                std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.dist < y.dist; });

                const int take = std::min(cfg::kTopNeighborsPerSide, static_cast<int>(cands.size()));

                // Best match for global high-level line
                if (take > 0) {
                    const auto best = cands[0];
                    const point2i p_from = all[a].side_centers_global[static_cast<std::size_t>(b)];
                    const point2i p_to = all[best.c].side_centers_global[static_cast<std::size_t>(best.d)];
                    draw_line_rgb(hi_best, p_from, p_to, /*R=*/255, /*G=*/0, /*B=*/0, /*thickness=*/1);
                }

                for (int t = 0; t < take; ++t) {
                    const auto nb = cands[static_cast<std::size_t>(t)];
                    const std::string C = idx4(nb.c);

                    std::ostringstream dist_ss;
                    dist_ss << std::fixed << std::setprecision(2) << nb.dist;

                    // Filename: A-B_neighb0_C-D_<distance>.png
                    std::ostringstream name;
                    name << A << "-" << b
                         << "_neighb" << t << "_"
                         << C << "-" << nb.d
                         << "_" << dist_ss.str() << ".png";

                    // Visualization: side B of object A and side D of object C side-by-side
                    const image8u left =
                        side_overlay_single(make_dark_rgb(objects[a].image), all[a].sides_pix[static_cast<std::size_t>(b)],
                                            cfg::kVizSeed ^ (std::uint32_t(a) * 131u + std::uint32_t(b) * 17u));

                    const image8u right =
                        side_overlay_single(make_dark_rgb(objects[nb.c].image),
                                            all[nb.c].sides_pix[static_cast<std::size_t>(nb.d)],
                                            cfg::kVizSeed ^ (std::uint32_t(nb.c) * 131u + std::uint32_t(nb.d) * 17u));

                    const image8u vis = concat_h(left, right);
                    libimages::debug_io::dump_image((dirB / name.str()).string(), vis, true, true);
                }
            }
        }

        libimages::debug_io::dump_image((out_dir / cfg::kOut04).string(), hi_best, true, true);

        std::cout << "objects: " << objects.size() << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
