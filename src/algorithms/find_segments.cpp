#include "find_segments.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <libbase/algorithms/disjoint_set.h>
#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>
#include <libbase/stats.h>

#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/algorithms/sobel_gradients.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

namespace find_segments {

namespace {

using libbase::point2i;
using libimages::image32f;
using libimages::image32i;
using libimages::image8u;

static std::string stage_path(const fs::path& out_dir, int step, const std::string& name, const std::string& ext) {
    const std::string filename = (step < 10 ? "0" : "") + std::to_string(step) + "_" + name + ext;
    return (out_dir / filename).string();
}

static std::string iters_dir_name(std::size_t unions_done) {
    std::ostringstream oss;
    oss << "iters" << std::setw(9) << std::setfill('0') << unions_done;
    return oss.str();
}

static std::vector<std::size_t> make_snapshot_schedule(std::size_t max_unions_possible) {
    std::vector<std::size_t> t;
    if (max_unions_possible >= 10) t.push_back(10);
    for (std::size_t x = 100; x <= 1000 && x <= max_unions_possible; x += 100) t.push_back(x);
    for (std::size_t x = 10000; x <= max_unions_possible; x *= 10) t.push_back(x);
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
}

static float median_f(const std::vector<float>& v) {
    rassert(!v.empty(), "median_f: empty vector");
    return static_cast<float>(stats::median(v));
}

static float median_abs_dev_f(const std::vector<float>& v, float med) {
    rassert(!v.empty(), "median_abs_dev_f: empty vector");
    std::vector<float> dev;
    dev.reserve(v.size());
    for (float x : v) dev.push_back(std::fabs(x - med));
    return median_f(dev);
}

static float center(float mn, float mx) { return 0.5f * (mn + mx); }

// Find k*360 shift that moves B close to A.
static float compute_shift_360(float centerA, float centerB) {
    const float k = std::round((centerA - centerB) / 360.0f);
    return k * 360.0f;
}

struct CompStats {
    std::vector<float> min_mag;
    std::vector<float> max_mag;
    std::vector<float> min_ang; // degrees, unwrapped
    std::vector<float> max_ang; // degrees, unwrapped
};

static bool can_merge_olson(const DisjointSetUnion& dsu, const CompStats& s, std::size_t ra, std::size_t rb,
                           float KD, float KM) {
    if (ra == rb) return false;

    const std::size_t sa = dsu.set_size(ra);
    const std::size_t sb = dsu.set_size(rb);
    const std::size_t sz = sa + sb;

    const float Da = s.max_ang[ra] - s.min_ang[ra];
    const float Db = s.max_ang[rb] - s.min_ang[rb];
    const float Ma = s.max_mag[ra] - s.min_mag[ra];
    const float Mb = s.max_mag[rb] - s.min_mag[rb];

    const float ca = center(s.min_ang[ra], s.max_ang[ra]);
    const float cb = center(s.min_ang[rb], s.max_ang[rb]);
    const float sh = compute_shift_360(ca, cb);

    const float minB = s.min_ang[rb] + sh;
    const float maxB = s.max_ang[rb] + sh;

    const float Dunion = std::max(s.max_ang[ra], maxB) - std::min(s.min_ang[ra], minB);
    const float Munion = std::max(s.max_mag[ra], s.max_mag[rb]) - std::min(s.min_mag[ra], s.min_mag[rb]);

    const float Dthr = std::min(Da, Db) + KD / static_cast<float>(sz);
    const float Mthr = std::min(Ma, Mb) + KM / static_cast<float>(sz);

    return (Dunion <= Dthr) && (Munion <= Mthr);
}

static void merge_stats_after_union(CompStats& s, std::size_t root_kept, std::size_t root_absorbed) {
    if (root_kept == root_absorbed) return;

    const float ca = center(s.min_ang[root_kept], s.max_ang[root_kept]);
    const float cb = center(s.min_ang[root_absorbed], s.max_ang[root_absorbed]);
    const float sh = compute_shift_360(ca, cb);

    const float minB = s.min_ang[root_absorbed] + sh;
    const float maxB = s.max_ang[root_absorbed] + sh;

    s.min_mag[root_kept] = std::min(s.min_mag[root_kept], s.min_mag[root_absorbed]);
    s.max_mag[root_kept] = std::max(s.max_mag[root_kept], s.max_mag[root_absorbed]);

    s.min_ang[root_kept] = std::min(s.min_ang[root_kept], minB);
    s.max_ang[root_kept] = std::max(s.max_ang[root_kept], maxB);
}

struct Edge {
    std::size_t a = 0;
    std::size_t b = 0;
    float w = 0.0f; // angle diff
};

static image32f make_image_component_size_log(const DisjointSetUnion& dsu, int w, int h) {
    image32f out(w, h, 1);
    out.fill(0.0f);

    std::size_t max_sz = 1;
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            max_sz = std::max(max_sz, dsu.set_size(static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i));

    const float max_log = std::log2(static_cast<float>(max_sz));
    const float denom = (max_log > 1e-6f) ? max_log : 1.0f;

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);
            const std::size_t sz = dsu.set_size(idx);
            if (sz <= 1) {
                out(j, i) = 0.0f;
                continue;
            }
            const float v = std::log2(static_cast<float>(sz)) / denom;
            out(j, i) = v * 255.0f;
        }
    }
    return out;
}

static image32f make_image_direction_range(const DisjointSetUnion& dsu, const CompStats& s, int w, int h) {
    image32f out(w, h, 1);
    out.fill(0.0f);

    float max_range = 0.0f;
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
            if (dsu.set_size(idx) <= 1) continue;
            const std::size_t r = dsu.find(idx);
            max_range = std::max(max_range, s.max_ang[r] - s.min_ang[r]);
        }
    }
    if (max_range < 1e-6f) max_range = 1.0f;

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
            if (dsu.set_size(idx) <= 1) continue;
            const std::size_t r = dsu.find(idx);
            const float range = s.max_ang[r] - s.min_ang[r];
            out(j, i) = (range / max_range) * 255.0f;
        }
    }
    return out;
}

static image32f make_image_magnitude_range(const DisjointSetUnion& dsu, const CompStats& s, int w, int h) {
    image32f out(w, h, 1);
    out.fill(0.0f);

    float max_range = 0.0f;
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
            if (dsu.set_size(idx) <= 1) continue;
            const std::size_t r = dsu.find(idx);
            max_range = std::max(max_range, s.max_mag[r] - s.min_mag[r]);
        }
    }
    if (max_range < 1e-6f) max_range = 1.0f;

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
            if (dsu.set_size(idx) <= 1) continue;
            const std::size_t r = dsu.find(idx);
            const float range = s.max_mag[r] - s.min_mag[r];
            out(j, i) = (range / max_range) * 255.0f;
        }
    }
    return out;
}

static image8u make_components_random_colors(const DisjointSetUnion& dsu, int w, int h, int min_size, std::uint32_t seed) {
    image8u out(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            out(j, i, 0) = out(j, i, 1) = out(j, i, 2) = 0;

    FastRandom rng(seed ? seed : 239U);
    std::unordered_map<std::size_t, std::array<std::uint8_t, 3>> color_of_root;
    color_of_root.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 8U + 16U);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);
            if (static_cast<int>(dsu.set_size(idx)) <= min_size) continue;

            const std::size_t r = dsu.find(idx);
            auto it = color_of_root.find(r);
            if (it == color_of_root.end()) {
                const std::uint32_t v = rng.nextU32();
                std::array<std::uint8_t, 3> c{
                    static_cast<std::uint8_t>(v & 0xFFU),
                    static_cast<std::uint8_t>((v >> 8) & 0xFFU),
                    static_cast<std::uint8_t>((v >> 16) & 0xFFU),
                };
                it = color_of_root.emplace(r, c).first;
            }

            out(j, i, 0) = it->second[0];
            out(j, i, 1) = it->second[1];
            out(j, i, 2) = it->second[2];
        }
    }
    return out;
}

static image8u make_component_mean_direction_hsv(const DisjointSetUnion& dsu, const CompStats& s, int w, int h) {
    image32f angle_deg(w, h, 1);
    image32f mag(w, h, 1);
    angle_deg.fill(0.0f);
    mag.fill(0.0f);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
            if (dsu.set_size(idx) <= 1) continue;

            const std::size_t r = dsu.find(idx);
            const float mean_unwrapped = center(s.min_ang[r], s.max_ang[r]);
            angle_deg(j, i) = libimages::wrap_angle_deg(mean_unwrapped);
            mag(j, i) = 1.0f; // constant brightness for non-singletons
        }
    }

    return libimages::visualize_angle_hsv(angle_deg, mag);
}

static void dump_unionfind_snapshot(const DebugParams& dbg, std::size_t unions_done,
                                    DisjointSetUnion& dsu, const CompStats& stats, int w, int h) {
    const fs::path dir = dbg.out_dir / iters_dir_name(unions_done);
    fs::create_directories(dir);

    libimages::debug_io::dump_image((dir / ("00_component_sizes_log" + dbg.dump_ext)).string(),
                                    make_image_component_size_log(dsu, w, h),
                                    dbg.verbose, dbg.force_overwrite);

    libimages::debug_io::dump_image((dir / ("01_direction_mean_hsv" + dbg.dump_ext)).string(),
                                    make_component_mean_direction_hsv(dsu, stats, w, h),
                                    dbg.verbose, dbg.force_overwrite);

    libimages::debug_io::dump_image((dir / ("02_direction_range" + dbg.dump_ext)).string(),
                                    make_image_direction_range(dsu, stats, w, h),
                                    dbg.verbose, dbg.force_overwrite);

    libimages::debug_io::dump_image((dir / ("03_magnitude_range" + dbg.dump_ext)).string(),
                                    make_image_magnitude_range(dsu, stats, w, h),
                                    dbg.verbose, dbg.force_overwrite);

    libimages::debug_io::dump_image((dir / ("04_components_random_colors" + dbg.dump_ext)).string(),
                                    make_components_random_colors(dsu, w, h, /*min_size=*/1, /*seed=*/239U),
                                    dbg.verbose, dbg.force_overwrite);

    libimages::debug_io::dump_image((dir / ("05_large_components_random_colors" + dbg.dump_ext)).string(),
                                    make_components_random_colors(dsu, w, h, dbg.large_component_min_size, /*seed=*/12345U),
                                    dbg.verbose, dbg.force_overwrite);
}

static image8u to_rgb_no_alpha(const image8u& img) {
    const int w = img.width();
    const int h = img.height();

    if (img.channels() == 3) return img;

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

    rassert(img.channels() == 4, "Unexpected channels in input image", img.channels());
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            out(j, i, 0) = img(j, i, 0);
            out(j, i, 1) = img(j, i, 1);
            out(j, i, 2) = img(j, i, 2);
        }

    return out;
}

} // namespace

std::vector<SegmentPixels> find_segments(const image8u& input, const Params& params, const DebugParams* debug) {
    rassert(input.width() > 0 && input.height() > 0, "Empty input image");

    const bool dbg_enabled = (debug != nullptr);
    if (dbg_enabled) fs::create_directories(debug->out_dir);

    if (dbg_enabled) {
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 0, "input", debug->dump_ext),
                                        input, debug->verbose, debug->force_overwrite);
    }

    // 01: grayscale (use library)
    const image32f gray = libimages::to_grayscale_float(input);
    if (dbg_enabled) {
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 1, "gray", debug->dump_ext),
                                        gray, debug->verbose, debug->force_overwrite);
    }

    // 02: blur (use library)
    const image32f blurred = libimages::gaussian_blur_gray(gray, params.gaussian_sigma);
    if (dbg_enabled) {
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 2, "gray_blurred", debug->dump_ext),
                                        blurred, debug->verbose, debug->force_overwrite);
    }

    // 03..: gradients (use library)
    const libimages::Gradients gr = libimages::sobel_gradients(blurred);
    const int w = gr.mag.width();
    const int h = gr.mag.height();

    if (dbg_enabled) {
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 3, "sobel_dx_signed", debug->dump_ext),
                                        libimages::visualize_signed_to_u8(gr.dx), debug->verbose, debug->force_overwrite);
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 4, "sobel_dy_signed", debug->dump_ext),
                                        libimages::visualize_signed_to_u8(gr.dy), debug->verbose, debug->force_overwrite);
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 5, "sobel_mag", debug->dump_ext),
                                        gr.mag, debug->verbose, debug->force_overwrite);
        libimages::debug_io::dump_image(stage_path(debug->out_dir, 6, "sobel_angle_hsv", debug->dump_ext),
                                        libimages::visualize_angle_hsv(gr.angle, gr.mag), debug->verbose, debug->force_overwrite);
    }

    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    DisjointSetUnion dsu(n);

    CompStats stats;
    stats.min_mag.resize(n);
    stats.max_mag.resize(n);
    stats.min_ang.resize(n);
    stats.max_ang.resize(n);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);
            const float a = libimages::wrap_angle_deg(gr.angle(j, i));
            const float m = gr.mag(j, i);
            stats.min_ang[idx] = a;
            stats.max_ang[idx] = a;
            stats.min_mag[idx] = m;
            stats.max_mag[idx] = m;
        }
    }

    // Build 8-connected edges
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>((w > 1 ? (w - 1) : 0) * h) +
                  static_cast<std::size_t>(w * (h > 1 ? (h - 1) : 0)) +
                  static_cast<std::size_t>(2 * (w > 1 ? (w - 1) : 0) * (h > 1 ? (h - 1) : 0)));

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::size_t a = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);

            if (i + 1 < w) {
                const std::size_t b = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i + 1);
                edges.push_back(Edge{a, b, libimages::angle_diff_deg(gr.angle(j, i), gr.angle(j, i + 1))});
            }
            if (j + 1 < h) {
                const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);
                edges.push_back(Edge{a, b, libimages::angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i))});
            }
            if (i + 1 < w && j + 1 < h) {
                const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i + 1);
                edges.push_back(Edge{a, b, libimages::angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i + 1))});
            }
            if (i - 1 >= 0 && j + 1 < h) {
                const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i - 1);
                edges.push_back(Edge{a, b, libimages::angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i - 1))});
            }
        }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) { return e1.w < e2.w; });

    const std::size_t max_unions_possible = (n > 0 ? (n - 1) : 0);
    const std::vector<std::size_t> schedule = dbg_enabled ? make_snapshot_schedule(max_unions_possible) : std::vector<std::size_t>{};
    std::size_t sched_pos = 0;
    std::size_t unions_done = 0;

    for (const Edge& e : edges) {
        const std::size_t ra = dsu.find(e.a);
        const std::size_t rb = dsu.find(e.b);
        if (ra == rb) continue;

        if (!can_merge_olson(dsu, stats, ra, rb, params.cluster_KD, params.cluster_KM)) continue;

        const auto [kept, absorbed] = dsu.unite_roots(ra, rb);
        if (kept == absorbed) continue;

        merge_stats_after_union(stats, kept, absorbed);
        ++unions_done;

        while (dbg_enabled && sched_pos < schedule.size() && unions_done == schedule[sched_pos]) {
            dump_unionfind_snapshot(*debug, unions_done, dsu, stats, w, h);
            ++sched_pos;
        }
    }

    // Build final segments
    std::vector<std::size_t> root_of(n);
    std::vector<int> count(n, 0);

    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::size_t r = dsu.find(idx);
        root_of[idx] = r;
        ++count[r];
    }

    std::vector<int> seg_id_of_root(n, -1);
    int seg_count = 0;
    for (std::size_t r = 0; r < n; ++r) {
        if (count[r] >= params.min_segment_pixels) seg_id_of_root[r] = seg_count++;
    }

    std::vector<SegmentPixels> segments(static_cast<std::size_t>(seg_count));
    std::vector<std::vector<float>> mags(static_cast<std::size_t>(seg_count));
    std::vector<std::vector<float>> ang_unwrapped(static_cast<std::size_t>(seg_count));
    std::vector<float> ref_center_unwrapped(static_cast<std::size_t>(seg_count), 0.0f);

    for (std::size_t r = 0; r < n; ++r) {
        const int sid = seg_id_of_root[r];
        if (sid < 0) continue;

        segments[static_cast<std::size_t>(sid)].pixels.reserve(static_cast<std::size_t>(count[r]));
        mags[static_cast<std::size_t>(sid)].reserve(static_cast<std::size_t>(count[r]));
        ang_unwrapped[static_cast<std::size_t>(sid)].reserve(static_cast<std::size_t>(count[r]));
        ref_center_unwrapped[static_cast<std::size_t>(sid)] = center(stats.min_ang[r], stats.max_ang[r]);
    }

    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::size_t r = root_of[idx];
        const int sid = seg_id_of_root[r];
        if (sid < 0) continue;

        const int y = static_cast<int>(idx / static_cast<std::size_t>(w));
        const int x = static_cast<int>(idx - static_cast<std::size_t>(y) * static_cast<std::size_t>(w));

        SegmentPixels& seg = segments[static_cast<std::size_t>(sid)];
        seg.pixels.push_back(point2i{x, y});

        const float m = gr.mag(y, x);
        mags[static_cast<std::size_t>(sid)].push_back(m);

        const float a = libimages::wrap_angle_deg(gr.angle(y, x));
        const float cref = ref_center_unwrapped[static_cast<std::size_t>(sid)];
        const float sh = compute_shift_360(cref, a);
        ang_unwrapped[static_cast<std::size_t>(sid)].push_back(a + sh);
    }

    for (std::size_t si = 0; si < segments.size(); ++si) {
        SegmentPixels& seg = segments[si];

        const float med_m = median_f(mags[si]);
        seg.median_magnitude = med_m;
        seg.median_abs_dev_magnitude = median_abs_dev_f(mags[si], med_m);

        // Median angle: median in unwrapped domain, then wrap back.
        const float med_unwrapped = median_f(ang_unwrapped[si]);
        const float med_a = libimages::wrap_angle_deg(med_unwrapped);
        seg.median_angle_deg = med_a;

        std::vector<float> dev;
        dev.reserve(seg.pixels.size());
        for (const point2i& p : seg.pixels) {
            const float a = libimages::wrap_angle_deg(gr.angle(p.y, p.x));
            dev.push_back(libimages::angle_diff_deg(a, med_a));
        }
        seg.median_abs_dev_angle_deg = median_f(dev);
    }

    return segments;
}

libimages::image8u visualize_segments_overlay(const image8u& input,
                                             const std::vector<SegmentPixels>& segments,
                                             const VisualizeParams& vis) {
    image8u out = to_rgb_no_alpha(input);

    // Darken background by 2x.
    for (int j = 0; j < out.height(); ++j) {
        for (int i = 0; i < out.width(); ++i) {
            out(j, i, 0) = static_cast<std::uint8_t>(out(j, i, 0) / 2);
            out(j, i, 1) = static_cast<std::uint8_t>(out(j, i, 1) / 2);
            out(j, i, 2) = static_cast<std::uint8_t>(out(j, i, 2) / 2);
        }
    }

    // Pre-generate random colors per segment (stable for given seed).
    FastRandom rng(vis.seed ? vis.seed : 239U);
    std::vector<std::array<std::uint8_t, 3>> seg_colors(segments.size());
    for (std::size_t si = 0; si < segments.size(); ++si) {
        const std::uint32_t v = rng.nextU32();
        seg_colors[si] = {
            static_cast<std::uint8_t>(v & 0xFFU),
            static_cast<std::uint8_t>((v >> 8) & 0xFFU),
            static_cast<std::uint8_t>((v >> 16) & 0xFFU),
        };
    }

    for (std::size_t si = 0; si < segments.size(); ++si) {
        const SegmentPixels& s = segments[si];

        if (static_cast<int>(s.pixels.size()) < vis.min_pixels) continue;
        if (s.median_magnitude < vis.min_median_magnitude) continue;
        if (s.median_abs_dev_angle_deg > vis.max_median_angle_deviation_deg) continue;

        const auto col = seg_colors[si];
        for (const point2i& p : s.pixels) {
            if (p.x < 0 || p.x >= out.width() || p.y < 0 || p.y >= out.height()) continue;
            out(p.y, p.x, 0) = col[0];
            out(p.y, p.x, 1) = col[1];
            out(p.y, p.x, 2) = col[2];
        }
    }

    return out;
}

} // namespace find_segments
