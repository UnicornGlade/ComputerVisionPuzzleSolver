#include "find_lines.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>

#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

namespace find_lines {

namespace {

constexpr float kPi = 3.14159265358979323846f;

static float clampf(float x, float a, float b) { return (x < a) ? a : (x > b) ? b : x; }

static float deg2rad(float d) { return d * kPi / 180.0f; }

static float angle_between_lines_deg(float ax, float ay, float bx, float by) {
    // Undirected line: angle between directions is acos(|dot|).
    const float dot = std::fabs(ax * bx + ay * by);
    const float d = clampf(dot, -1.0f, 1.0f);
    return std::acos(d) * 180.0f / kPi;
}

static float point_line_distance_px(float x, float y, float x0, float y0, float ux, float uy) {
    // Distance to infinite line through (x0,y0) with unit direction (ux,uy):
    // |(p - p0) x u| = |(x-x0)*uy - (y-y0)*ux|
    return std::fabs((x - x0) * uy - (y - y0) * ux);
}

struct LineModel {
    point2i medoid{0, 0};
    float ux = 1.0f;
    float uy = 0.0f;
};

// Compute centroid of pixel centers and choose medoid pixel (closest to centroid).
static point2i compute_medoid(const std::vector<point2i>& px) {
    rassert(!px.empty(), "compute_medoid: empty segment");

    double sx = 0.0;
    double sy = 0.0;
    for (const auto& p : px) {
        sx += static_cast<double>(p.x) + 0.5;
        sy += static_cast<double>(p.y) + 0.5;
    }
    const double cx = sx / static_cast<double>(px.size());
    const double cy = sy / static_cast<double>(px.size());

    double best_d2 = std::numeric_limits<double>::infinity();
    point2i best = px[0];

    for (const auto& p : px) {
        const double x = static_cast<double>(p.x) + 0.5;
        const double y = static_cast<double>(p.y) + 0.5;
        const double dx = x - cx;
        const double dy = y - cy;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = p;
        }
    }
    return best;
}

// Direction is perpendicular to median gradient angle.
// If grad angle is a, line angle is a + 90.
static LineModel build_line_model_from_segment(const find_segments::SegmentPixels& s) {
    LineModel m;
    m.medoid = compute_medoid(s.pixels);

    const float a = deg2rad(s.median_angle_deg + 90.0f);
    m.ux = std::cos(a);
    m.uy = std::sin(a);

    const float n = std::sqrt(m.ux * m.ux + m.uy * m.uy);
    if (n > 1e-12f) {
        m.ux /= n;
        m.uy /= n;
    } else {
        m.ux = 1.0f;
        m.uy = 0.0f;
    }
    return m;
}

static bool segment_votes_for_line(const find_segments::SegmentPixels& seg,
                                  const LineModel& seg_line,
                                  const LineModel& cand,
                                  float max_angle_deg,
                                  float max_dist_px) {
    // Angle check (line-line).
    const float a = angle_between_lines_deg(seg_line.ux, seg_line.uy, cand.ux, cand.uy);
    if (a > max_angle_deg)
        return false;

    // Distance check: at least half of pixels within max_dist_px to candidate line.
    const float x0 = static_cast<float>(cand.medoid.x) + 0.5f;
    const float y0 = static_cast<float>(cand.medoid.y) + 0.5f;

    const std::size_t need = (seg.pixels.size() + 1) / 2;
    std::size_t ok = 0;

    for (const auto& p : seg.pixels) {
        const float x = static_cast<float>(p.x) + 0.5f;
        const float y = static_cast<float>(p.y) + 0.5f;
        const float d = point_line_distance_px(x, y, x0, y0, cand.ux, cand.uy);
        if (d <= max_dist_px) {
            ++ok;
            if (ok >= need)
                return true;
        }
    }
    return false;
}

} // namespace

std::vector<LinePixels> find_lines(const std::vector<find_segments::SegmentPixels>& segments, const Params& params) {
    rassert(params.min_candidate_pixels > 0, "min_candidate_pixels must be > 0", params.min_candidate_pixels);
    rassert(params.max_iterations > 0, "max_iterations must be > 0", params.max_iterations);
    rassert(params.max_angle_diff_deg >= 0.0f, "max_angle_diff_deg must be >= 0", params.max_angle_diff_deg);
    rassert(params.max_dist_px >= 0.0f, "max_dist_px must be >= 0", params.max_dist_px);
    rassert(params.min_total_votes_pixels >= 0, "min_total_votes_pixels must be >= 0", params.min_total_votes_pixels);

    const std::size_t n = segments.size();
    std::vector<bool> active(n, true);

    // Precompute per-segment line models (medoid + dir).
    std::vector<LineModel> seg_line(n);
    std::vector<bool> is_candidate(n, false);

    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<int>(segments[i].pixels.size()) >= params.min_candidate_pixels) {
            seg_line[i] = build_line_model_from_segment(segments[i]);
            is_candidate[i] = true;
        } else {
            is_candidate[i] = false;
        }
    }

    std::vector<LinePixels> out;
    out.reserve(64);

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        // Collect current candidate indices.
        std::vector<std::size_t> candidates;
        candidates.reserve(n);
        std::vector<std::size_t> voters;
        voters.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            if (!active[i]) continue;
            voters.push_back(i);
            if (is_candidate[i]) candidates.push_back(i);
        }

        if (candidates.empty())
            break;

        int best_votes = -1;
        std::size_t best_cand = candidates[0];
        std::vector<std::size_t> best_inliers;

        // For each candidate line, compute votes from all active segments.
        for (std::size_t ci : candidates) {
            const LineModel& cand = seg_line[ci];

            int votes = 0;
            std::vector<std::size_t> inliers;
            inliers.reserve(256);

            for (std::size_t si : voters) {
                const bool ok = segment_votes_for_line(segments[si], seg_line[si], cand,
                                                      params.max_angle_diff_deg, params.max_dist_px);
                if (!ok) continue;

                const int w = static_cast<int>(segments[si].pixels.size());
                votes += w;
                inliers.push_back(si);
            }

            if (votes > best_votes) {
                best_votes = votes;
                best_cand = ci;
                best_inliers = std::move(inliers);
            }
        }

        if (best_votes < params.min_total_votes_pixels)
            break;

        // Emit output line = union of pixels of all inlier segments.
        LinePixels line;
        line.medoid = seg_line[best_cand].medoid;
        line.dir_x = seg_line[best_cand].ux;
        line.dir_y = seg_line[best_cand].uy;
        line.total_votes_pixels = best_votes;
        line.inlier_segments = static_cast<int>(best_inliers.size());

        std::size_t total_px = 0;
        for (std::size_t si : best_inliers) total_px += segments[si].pixels.size();
        line.pixels.reserve(total_px);

        for (std::size_t si : best_inliers) {
            const auto& px = segments[si].pixels;
            line.pixels.insert(line.pixels.end(), px.begin(), px.end());
        }

        out.push_back(std::move(line));

        // Remove inlier segments from further iterations (both as candidates and voters).
        for (std::size_t si : best_inliers) active[si] = false;
    }

    return out;
}

libimages::image8u visualize_lines_overlay(const libimages::image8u& input,
                                          const std::vector<LinePixels>& lines,
                                          const VisualizeParams& vis) {
    // Convert input to RGB (no alpha), then darken 2x and paint pixels.
    libimages::image8u out(input.width(), input.height(), 3);

    if (input.channels() == 3) {
        out = input;
    } else if (input.channels() == 1) {
        for (int j = 0; j < out.height(); ++j)
            for (int i = 0; i < out.width(); ++i) {
                const std::uint8_t v = input(j, i);
                out(j, i, 0) = v;
                out(j, i, 1) = v;
                out(j, i, 2) = v;
            }
    } else if (input.channels() == 4) {
        for (int j = 0; j < out.height(); ++j)
            for (int i = 0; i < out.width(); ++i) {
                out(j, i, 0) = input(j, i, 0);
                out(j, i, 1) = input(j, i, 1);
                out(j, i, 2) = input(j, i, 2);
            }
    } else {
        rassert(false, "Unsupported input channels for visualize_lines_overlay", input.channels());
    }

    for (int j = 0; j < out.height(); ++j)
        for (int i = 0; i < out.width(); ++i) {
            out(j, i, 0) = static_cast<std::uint8_t>(out(j, i, 0) / 2);
            out(j, i, 1) = static_cast<std::uint8_t>(out(j, i, 1) / 2);
            out(j, i, 2) = static_cast<std::uint8_t>(out(j, i, 2) / 2);
        }

    // Stable random colors via RNG stream.
    FastRandom rng(vis.seed ? vis.seed : 239U);
    std::vector<std::array<std::uint8_t, 3>> colors(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::uint32_t v = rng.nextU32();
        colors[i] = {static_cast<std::uint8_t>(v & 0xFFU),
                     static_cast<std::uint8_t>((v >> 8) & 0xFFU),
                     static_cast<std::uint8_t>((v >> 16) & 0xFFU)};
    }

    for (std::size_t li = 0; li < lines.size(); ++li) {
        const auto& ln = lines[li];
        if (static_cast<int>(ln.pixels.size()) < vis.min_pixels)
            continue;

        const auto col = colors[li];
        for (const auto& p : ln.pixels) {
            if (p.x < 0 || p.x >= out.width() || p.y < 0 || p.y >= out.height())
                continue;
            out(p.y, p.x, 0) = col[0];
            out(p.y, p.x, 1) = col[1];
            out(p.y, p.x, 2) = col[2];
        }
    }

    return out;
}

} // namespace find_lines
