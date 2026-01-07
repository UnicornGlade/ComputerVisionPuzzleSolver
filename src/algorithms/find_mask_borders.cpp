#include "find_mask_borders.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <libbase/algorithms/disjoint_set.h>
#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/sobel_gradients.h>
#include <libimages/debug_io.h>

namespace find_mask_borders {

static inline bool is_fg(std::uint8_t v) { return v == 255; }

static void check_mask01(const libimages::image8u& m) {
    rassert(m.channels() == 1, "mask must be 1-channel", m.channels());
    for (int j = 0; j < m.height(); ++j)
        for (int i = 0; i < m.width(); ++i) {
            const auto v = m(j, i);
            rassert(v == 0 || v == 255, "mask must contain only 0/255", int(v), j, i);
        }
}

static libimages::image8u to_rgb_darken(const libimages::image8u& img, std::uint8_t div) {
    rassert(div >= 1, "div must be >= 1", int(div));
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported channels", img.channels());

    libimages::image8u out(img.width(), img.height(), 3);
    for (int j = 0; j < img.height(); ++j) {
        for (int i = 0; i < img.width(); ++i) {
            if (img.channels() == 1) {
                const std::uint8_t v = static_cast<std::uint8_t>(img(j, i) / div);
                out(j, i, 0) = v;
                out(j, i, 1) = v;
                out(j, i, 2) = v;
            } else {
                out(j, i, 0) = static_cast<std::uint8_t>(img(j, i, 0) / div);
                out(j, i, 1) = static_cast<std::uint8_t>(img(j, i, 1) / div);
                out(j, i, 2) = static_cast<std::uint8_t>(img(j, i, 2) / div);
            }
        }
    }
    return out;
}

static point3u get_rgb(const libimages::image8u& img, point2i p) {
    rassert(img.channels() == 1 || img.channels() == 3 || img.channels() == 4, "Unsupported image channels",
            img.channels());
    if (img.channels() == 1) {
        const auto v = img(p.y, p.x);
        return point3u{v, v, v};
    }
    return point3u{img(p.y, p.x, 0), img(p.y, p.x, 1), img(p.y, p.x, 2)};
}

static libimages::image8u gray32f_to_u8_clamp_local(const libimages::image32f& gray) {
    rassert(gray.channels() == 1, "gray32f_to_u8_clamp_local expects 1-channel image32f", gray.channels());
    libimages::image8u out(gray.width(), gray.height(), 1);
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

static void dump_dbg(const SplitSidesDebugParams* dbg, const std::string& name, const libimages::image8u& img) {
    if (!dbg) return;
    if (dbg->out_dir.empty()) return;
    std::filesystem::create_directories(dbg->out_dir);
    libimages::debug_io::dump_image((dbg->out_dir / (dbg->prefix + name + dbg->dump_ext)).string(),
                                    img, dbg->verbose, dbg->force_overwrite);
}

static void dump_dbg32_as_u8(const SplitSidesDebugParams* dbg, const std::string& name, const libimages::image32f& img) {
    if (!dbg) return;
    dump_dbg(dbg, name, gray32f_to_u8_clamp_local(img));
}

static void put_pixel_thick(libimages::image8u& img, int x, int y, point3u c, int thickness) {
    // thickness=1 => single pixel, thickness=2 => 3x3, thickness=3 => 5x5, etc.
    rassert(img.channels() == 3, "put_pixel_thick expects 3-channel image", img.channels());

    const int w = img.width();
    const int h = img.height();

    if (x < 0 || x >= w || y < 0 || y >= h) return;

    const int rad = std::max(0, thickness - 1);
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            const int xx = x + dx;
            const int yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            img(yy, xx, 0) = c.r;
            img(yy, xx, 1) = c.g;
            img(yy, xx, 2) = c.b;
        }
    }
}

static void put_point_rgb(libimages::image8u& img, int x, int y, point3u c, int rad) {
    const int w = img.width();
    const int h = img.height();
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            const int xx = x + dx;
            const int yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            img(yy, xx, 0) = c.r;
            img(yy, xx, 1) = c.g;
            img(yy, xx, 2) = c.b;
        }
    }
}

// This is the one you asked for (Bresenham, like your example)
static void draw_line_rgb(libimages::image8u& img, point2i a, point2i b, point3u c, int thickness) {
    rassert(img.channels() == 3, "draw_line_rgb expects 3-channel image", img.channels());
    thickness = std::max(1, thickness);

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
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void draw_rect_rgb_inplace(libimages::image8u& rgb, int x0, int y0, int x1, int y1,
                                  point3u col, int thick) {
    // Draw rectangle border with Bresenham lines (using your existing draw_line_rgb helper).
    for (int t = 0; t < thick; ++t) {
        draw_line_rgb(rgb, point2i{x0 - t, y0 - t}, point2i{x1 + t, y0 - t}, col, 0);
        draw_line_rgb(rgb, point2i{x1 + t, y0 - t}, point2i{x1 + t, y1 + t}, col, 0);
        draw_line_rgb(rgb, point2i{x1 + t, y1 + t}, point2i{x0 - t, y1 + t}, col, 0);
        draw_line_rgb(rgb, point2i{x0 - t, y1 + t}, point2i{x0 - t, y0 - t}, col, 0);
    }
}

static libimages::image8u visualize_grad_arrows_rgb(const libimages::image8u& base_rgb,
                                                    const libimages::Gradients& g,
                                                    float min_mag,
                                                    int stride,
                                                    float arrow_len,
                                                    int thick) {
    libimages::image8u out = base_rgb; // copy

    const int w = out.width();
    const int h = out.height();

    for (int y = 0; y < h; y += std::max(1, stride)) {
        for (int x = 0; x < w; x += std::max(1, stride)) {
            const float mag = g.mag(y, x);
            if (!(mag > min_mag)) continue;

            const float dx = g.dx(y, x);
            const float dy = g.dy(y, x);
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len < 1e-6f) continue;

            const float ux = dx / len;
            const float uy = dy / len;

            const int x1 = static_cast<int>(std::lround(static_cast<float>(x) + arrow_len * ux));
            const int y1 = static_cast<int>(std::lround(static_cast<float>(y) + arrow_len * uy));

            draw_line_rgb(out, point2i{x, y}, point2i{x1, y1}, point3u{255, 0, 0}, std::max(0, thick - 1));
        }
    }

    return out;
}



libimages::image8u border_pixels_mask(const libimages::image8u& object_mask, int min_bg_neighbors) {
    rassert(min_bg_neighbors >= 0 && min_bg_neighbors <= 8, "min_bg_neighbors must be in [0,8]", min_bg_neighbors);
    check_mask01(object_mask);

    const int w = object_mask.width();
    const int h = object_mask.height();

    libimages::image8u out(w, h, 1);
    out.fill(static_cast<std::uint8_t>(0));

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (!is_fg(object_mask(j, i))) continue;

            int bg = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int yy = j + dy;
                    const int xx = i + dx;
                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
                        bg += 1;
                    } else {
                        bg += (object_mask(yy, xx) == 0) ? 1 : 0;
                    }
                }
            }
            if (bg >= min_bg_neighbors) out(j, i) = 255;
        }
    }
    return out;
}

struct Edge final {
    int a = 0;        // linear id in [0, w*h)
    int b = 0;
    float ang_diff = 0.0f;
};

static inline int lin_id(int x, int y, int w) { return y * w + x; }
static inline point2i lin_id_reverse(int id, int w) { return {id % w, id / w}; }

static void compute_object_centroid(const libimages::image8u& object_mask, double& cx, double& cy) {
    double sx = 0.0, sy = 0.0;
    double cnt = 0.0;
    for (int j = 0; j < object_mask.height(); ++j) {
        for (int i = 0; i < object_mask.width(); ++i) {
            if (object_mask(j, i) != 255) continue;
            sx += i;
            sy += j;
            cnt += 1.0;
        }
    }
    if (cnt <= 0.0) {
        cx = 0.0;
        cy = 0.0;
    } else {
        cx = sx / cnt;
        cy = sy / cnt;
    }
}

static float angle_mean_deg_from_pixels(const libimages::image32f& angle_deg_img,
                                       const std::vector<point2i>& pix) {
    double sx = 0.0, sy = 0.0;
    for (const auto& p : pix) {
        const double a = static_cast<double>(angle_deg_img(p.y, p.x)) * (3.14159265358979323846 / 180.0);
        sx += std::cos(a);
        sy += std::sin(a);
    }
    if (pix.empty()) return 0.0f;
    const double a = std::atan2(sy, sx) * (180.0 / 3.14159265358979323846);
    float deg = static_cast<float>(a);
    if (deg < 0.0f) deg += 360.0f;
    return deg;
}

std::vector<std::vector<point2i>> split_border_into_sides(const libimages::image8u& object_mask,
                                                         const libimages::image8u& border_mask,
                                                         int side_count,
                                                         SplitSidesDebugParams* dbg) {
    rassert(side_count >= 1, "side_count must be >= 1", side_count);
    check_mask01(object_mask);
    check_mask01(border_mask);
    rassert(object_mask.width() == border_mask.width() && object_mask.height() == border_mask.height(),
            "object_mask and border_mask must have same size",
            object_mask.width(), object_mask.height(), border_mask.width(), border_mask.height());

    const int w = object_mask.width();
    const int h = object_mask.height();

    const int pad = (dbg ? dbg->pad : 10);
    const float blur_sigma = (dbg ? dbg->blur_sigma : 0.5f);

    const int wp = w + 2 * pad;
    const int hp = h + 2 * pad;

    // Padded float image for Sobel.
    libimages::image32f m(wp, hp, 1);
    m.fill(0.0f);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            m(j + pad, i + pad) = static_cast<float>(object_mask(j, i));

    // Debug: padded mask before blur
    if (dbg && !dbg->out_dir.empty()) {
        libimages::image8u m_u8(wp, hp, 1);
        for (int y = 0; y < hp; ++y)
            for (int x = 0; x < wp; ++x)
                m_u8(y, x) = (m(y, x) > 0.0f) ? 255 : 0;

        auto base = to_rgb_darken(m_u8, 1);
        draw_rect_rgb_inplace(base, pad, pad, pad + w - 1, pad + h - 1, point3u{255, 255, 0}, 1);
        dump_dbg(dbg, "pad00_mask_roi", base);
    }

    // Blur before Sobel (padded)
    m = libimages::gaussian_blur_gray(m, blur_sigma);
    if (dbg && !dbg->out_dir.empty()) {
        auto blur_u8 = gray32f_to_u8_clamp_local(m);
        auto blur_rgb = to_rgb_darken(blur_u8, 1);
        draw_rect_rgb_inplace(blur_rgb, pad, pad, pad + w - 1, pad + h - 1, point3u{255, 255, 0}, 1);
        dump_dbg(dbg, "pad01_blur_roi", blur_rgb);
    }

    const libimages::Gradients g = libimages::sobel_gradients(m);

    // Debug: angle HSV + arrows (both are padded-sized)
    if (dbg && !dbg->out_dir.empty()) {
        auto hsv = libimages::visualize_angle_hsv(g.angle, g.mag);
        draw_rect_rgb_inplace(hsv, pad, pad, pad + w - 1, pad + h - 1, point3u{255, 255, 0}, 1);
        dump_dbg(dbg, "pad02_angle_hsv", hsv);

        auto blur_u8 = gray32f_to_u8_clamp_local(m);
        auto base_rgb = to_rgb_darken(blur_u8, 1);
        draw_rect_rgb_inplace(base_rgb, pad, pad, pad + w - 1, pad + h - 1, point3u{255, 255, 0}, 1);

        const int stride = dbg->arrow_stride;
        const float min_mag = dbg->arrow_min_mag;
        const float arrow_len = dbg->arrow_len_px;
        const int thick = dbg->arrow_thickness;

        auto arrows = visualize_grad_arrows_rgb(base_rgb, g, min_mag, stride, arrow_len, thick);
        dump_dbg(dbg, "pad03_grad_arrows", arrows);

        dump_dbg(dbg, "pad04_dx_signed_u8", libimages::visualize_signed_to_u8(g.dx));
        dump_dbg(dbg, "pad05_dy_signed_u8", libimages::visualize_signed_to_u8(g.dy));
        dump_dbg32_as_u8(dbg, "pad06_mag_u8", g.mag);
    }

    // Read padded gradients using original coordinates
    auto angle_at = [&](int y, int x) -> float { return g.angle(y + pad, x + pad); };

    // Collect border ids (un-padded)
    const int N = w * h;
    std::vector<int> id_to_idx(static_cast<std::size_t>(N), -1);
    std::vector<int> border_ids;
    border_ids.reserve(1024);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (border_mask(j, i) != 255) continue;
            const int id = lin_id(i, j, w);
            id_to_idx[static_cast<std::size_t>(id)] = static_cast<int>(border_ids.size());
            border_ids.push_back(id);
        }
    }
    if (border_ids.empty()) return {};

    // Build adjacency edges (8-neighborhood, unique via forward directions).
    std::vector<Edge> edges;
    edges.reserve(border_ids.size() * 4);

    auto add_edge_if = [&](int x0, int y0, int x1, int y1) {
        if (x1 < 0 || x1 >= w || y1 < 0 || y1 >= h) return;
        if (border_mask(y0, x0) != 255) return;
        if (border_mask(y1, x1) != 255) return;

        const float a0 = angle_at(y0, x0);
        const float a1 = angle_at(y1, x1);
        const float d = libimages::angle_diff_deg(a0, a1);

        edges.push_back(Edge{lin_id(x0, y0, w), lin_id(x1, y1, w), d});
    };

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (border_mask(j, i) != 255) continue;
            add_edge_if(i, j, i + 1, j);     // right
            add_edge_if(i, j, i, j + 1);     // down
            add_edge_if(i, j, i + 1, j + 1); // down-right
            add_edge_if(i, j, i - 1, j + 1); // down-left
        }
    }

        // Compact edges to DSU indices once (and keep pixel coords for debug).
    struct EdgeC final {
        int ia = -1;
        int ib = -1;
        float ang = 0.0f;
        point2i pa{};
        point2i pb{};
    };

    std::vector<EdgeC> edges_c;
    edges_c.reserve(edges.size());

    for (const auto& e : edges) {
        const int ia = id_to_idx[static_cast<std::size_t>(e.a)];
        const int ib = id_to_idx[static_cast<std::size_t>(e.b)];
        if (ia < 0 || ib < 0) continue;

        const int ax = e.a % w;
        const int ay = e.a / w;
        const int bx = e.b % w;
        const int by = e.b / w;

        // Sanity: must be 8-neighbors (for debugging confidence)
        rassert(std::abs(ax - bx) <= 1 && std::abs(ay - by) <= 1, "Non-neighbor edge unexpectedly",
                ax, ay, bx, by);

        edges_c.push_back(EdgeC{ia, ib, e.ang_diff, point2i{ax, ay}, point2i{bx, by}});
    }

    DisjointSetUnion dsu(border_ids.size());
    int comps = static_cast<int>(border_ids.size());

    constexpr float kBinStepDeg = 15.0f;
    constexpr float kMaxDiffDeg = 180.0f;
    const int nbins = static_cast<int>(std::ceil(kMaxDiffDeg / kBinStepDeg));

    // Bucket edges by [0,15), [15,30), ...
    std::vector<std::vector<int>> bins(static_cast<std::size_t>(nbins));
    bins.shrink_to_fit(); // keeps it simple to view in debugger

    for (int ei = 0; ei < static_cast<int>(edges_c.size()); ++ei) {
        float d = edges_c[static_cast<std::size_t>(ei)].ang;
        d = std::clamp(d, 0.0f, kMaxDiffDeg);
        int b = static_cast<int>(d / kBinStepDeg);
        if (b >= nbins) b = nbins - 1;
        bins[static_cast<std::size_t>(b)].push_back(ei);
    }

    auto comp_size = [&](int idx) -> std::size_t {
        return dsu.set_size(static_cast<std::size_t>(idx));
    };

    // Merge bin-by-bin. Inside bin: greedy pick best edge among eligible in this bin.
    for (int b = 0; b < nbins && comps > side_count; ++b) {
        auto& bin = bins[static_cast<std::size_t>(b)];
        if (bin.empty()) continue;

        while (comps > side_count) {
            int best_ei = -1;
            double best_score = -1.0;     // maximize ratio max/min
            std::size_t best_max = 0;     // tie-breaker
            float best_ang = 1e9f;        // tie-breaker: smaller ang

            for (int kk = 0; kk < static_cast<int>(bin.size()); ++kk) {
                const int ei = bin[static_cast<std::size_t>(kk)];
                const auto& e = edges_c[static_cast<std::size_t>(ei)];

                const std::size_t ra = dsu.find(static_cast<std::size_t>(e.ia));
                const std::size_t rb = dsu.find(static_cast<std::size_t>(e.ib));
                if (ra == rb) continue;

                const std::size_t sa = dsu.set_size(static_cast<std::size_t>(e.ia));
                const std::size_t sb = dsu.set_size(static_cast<std::size_t>(e.ib));

                const std::size_t mn = std::min(sa, sb);
                const std::size_t mx = std::max(sa, sb);

                // sizes are always >=1, but keep robust
                const double score = (mn > 0) ? (static_cast<double>(mx) / static_cast<double>(mn)) : 1e300;

                // primary: maximize ratio (big+small)
                // tie1: maximize mx (prefer larger big set)
                // tie2: minimize ang
                if (best_ei < 0 ||
                    score > best_score ||
                    (score == best_score && mx > best_max) ||
                    (score == best_score && mx == best_max && e.ang < best_ang)) {
                    best_ei = ei;
                    best_score = score;
                    best_max = mx;
                    best_ang = e.ang;
                }
            }

            if (best_ei < 0) break; // no eligible merges in this bin

            const auto& e = edges_c[static_cast<std::size_t>(best_ei)];

            // === IMPORTANT DEBUG: print exact pixels + sizes ===
            if (dbg && dbg->verbose) {
                const std::size_t sa = dsu.set_size(static_cast<std::size_t>(e.ia));
                const std::size_t sb = dsu.set_size(static_cast<std::size_t>(e.ib));
                const std::size_t mn = std::min(sa, sb);
                const std::size_t mx = std::max(sa, sb);
                const double sc = (mn > 0) ? (static_cast<double>(mx) / static_cast<double>(mn)) : 1e300;

                std::cerr << "unite: (" << e.pa.x << "," << e.pa.y << ") <-> (" << e.pb.x << "," << e.pb.y << ")"
                          << " ang=" << e.ang
                          << " sizes=(" << sa << "," << sb << ")"
                          << " score=" << sc
                          << "\n";
            }

            if (dsu.unite(static_cast<std::size_t>(e.ia), static_cast<std::size_t>(e.ib))) {
                comps -= 1;
            }
        }
    }

    // If still too many components (graph disconnected / too sparse), keep biggest side_count later.
    // --- end NEW MERGE LOGIC ---

    // Gather components.
    std::vector<int> root_to_comp(border_ids.size(), -1);
    std::vector<std::vector<point2i>> groups;

    for (int idx = 0; idx < static_cast<int>(border_ids.size()); ++idx) {
        const std::size_t r = dsu.find(static_cast<std::size_t>(idx));
        int& slot = root_to_comp[static_cast<std::size_t>(r)];
        if (slot < 0) {
            slot = static_cast<int>(groups.size());
            groups.emplace_back();
        }
        const int id = border_ids[static_cast<std::size_t>(idx)];
        const int x = id % w;
        const int y = id / w;
        groups[static_cast<std::size_t>(slot)].push_back(point2i{x, y});
    }

    if (static_cast<int>(groups.size()) > side_count) {
        std::sort(groups.begin(), groups.end(),
                  [](const auto& a, const auto& b) { return a.size() > b.size(); });
        groups.resize(static_cast<std::size_t>(side_count));
    }

    // Order sides around object centroid (clockwise).
    double ocx = 0.0, ocy = 0.0;
    compute_object_centroid(object_mask, ocx, ocy);

    struct SideMeta { std::vector<point2i> pix; double ang = 0.0; };
    std::vector<SideMeta> meta;
    meta.reserve(groups.size());

    for (auto& s : groups) {
        double sx = 0.0, sy = 0.0;
        for (const auto& p : s) { sx += p.x; sy += p.y; }
        const double cx = s.empty() ? 0.0 : sx / static_cast<double>(s.size());
        const double cy = s.empty() ? 0.0 : sy / static_cast<double>(s.size());

        const double dx = cx - ocx;
        const double dy = cy - ocy;
        double ang = std::atan2(-dy, dx);
        if (ang < 0.0) ang += 2.0 * 3.14159265358979323846;

        meta.push_back(SideMeta{std::move(s), ang});
    }

    std::sort(meta.begin(), meta.end(), [](const SideMeta& a, const SideMeta& b) { return a.ang < b.ang; });

    // Mean inward angle for side pixels using padded gradients.
    auto mean_inward_deg_for_side = [&](const std::vector<point2i>& pix) -> float {
        if (pix.empty()) return 0.0f;
        double sx = 0.0, sy = 0.0;
        for (const auto& p : pix) {
            const double deg = static_cast<double>(angle_at(p.y, p.x));
            const double a = deg * (3.14159265358979323846 / 180.0);
            sx += std::cos(a);
            sy += std::sin(a);
        }
        double ang = std::atan2(sy, sx) * (180.0 / 3.14159265358979323846);
        if (ang < 0.0) ang += 360.0;
        return static_cast<float>(ang);
    };

    // Order pixels within each side approximately clockwise along the side using tangent = rotate(inward, +90).
    std::vector<std::vector<point2i>> out;
    out.reserve(meta.size());

    for (auto& sm : meta) {
        auto& s = sm.pix;
        if (s.size() <= 1) { out.push_back(std::move(s)); continue; }

        const float mean_inward_deg = mean_inward_deg_for_side(s);
        const double a = static_cast<double>(mean_inward_deg) * (3.14159265358979323846 / 180.0);
        const double nx = std::cos(a);
        const double ny = std::sin(a);

        // tangent direction for clockwise traversal: t = rotate(inward, +90) = (-ny, nx)
        const double tx = -ny;
        const double ty = nx;

        std::sort(s.begin(), s.end(), [&](const point2i& p1, const point2i& p2) {
            const double pr1 = p1.x * tx + p1.y * ty;
            const double pr2 = p2.x * tx + p2.y * ty;
            if (pr1 != pr2) return pr1 < pr2;
            const double or1 = p1.x * nx + p1.y * ny;
            const double or2 = p2.x * nx + p2.y * ny;
            return or1 < or2;
        });

        out.push_back(std::move(s));
    }

    return out;
}

std::vector<std::vector<point3u>> sample_sides_colors(const libimages::image8u& image_rgb_or_gray,
                                                      const std::vector<std::vector<point2i>>& sides_clockwise,
                                                      int samples_per_side,
                                                      SamplingDebugParams* dbg) {
    rassert(samples_per_side >= 1, "samples_per_side must be >= 1", samples_per_side);
    rassert(image_rgb_or_gray.channels() == 1 || image_rgb_or_gray.channels() == 3 || image_rgb_or_gray.channels() == 4,
            "Unsupported image channels", image_rgb_or_gray.channels());

    const int w = image_rgb_or_gray.width();
    const int h = image_rgb_or_gray.height();

    std::vector<std::vector<point3u>> out;
    out.reserve(sides_clockwise.size());

    const bool do_dbg = (dbg != nullptr) && !dbg->out_dir.empty();
    if (do_dbg) std::filesystem::create_directories(dbg->out_dir);

    // Combined debug overlay (all sides)
    libimages::image8u dbg_all;
    if (do_dbg) dbg_all = to_rgb_darken(image_rgb_or_gray, 2);

    for (std::size_t si = 0; si < sides_clockwise.size(); ++si) {
        const auto& side = sides_clockwise[si];
        std::vector<point3u> colors;
        colors.reserve(static_cast<std::size_t>(samples_per_side));

        if (side.empty()) {
            out.push_back(colors);
            continue;
        }
        if (side.size() == 1) {
            // Degenerate side: sample same pixel.
            const point3u c = get_rgb(image_rgb_or_gray, side[0]);
            for (int k = 0; k < samples_per_side; ++k) colors.push_back(c);
            out.push_back(colors);
            continue;
        }

        const point2i a = side.front();
        const point2i b = side.back();

        const double vx = static_cast<double>(b.x - a.x);
        const double vy = static_cast<double>(b.y - a.y);
        const double len = std::sqrt(vx * vx + vy * vy);
        const double ux = (len > 1e-9) ? (vx / len) : 1.0;
        const double uy = (len > 1e-9) ? (vy / len) : 0.0;

        // Prepare donors sorted by projection s along line.
        struct Donor {
            double s;
            point2i p;
        };
        std::vector<Donor> donors;
        donors.reserve(side.size());
        for (const auto& p : side) {
            const double dx = static_cast<double>(p.x - a.x);
            const double dy = static_cast<double>(p.y - a.y);
            const double s = dx * ux + dy * uy;
            donors.push_back(Donor{s, p});
        }
        std::sort(donors.begin(), donors.end(), [](const Donor& d1, const Donor& d2) { return d1.s < d2.s; });

        // Per-side debug overlay
        libimages::image8u dbg_side;
        if (do_dbg) {
            dbg_side = to_rgb_darken(image_rgb_or_gray, 2);
            // draw side chord in light gray
            draw_line_rgb(dbg_side, a, b, point3u{200, 200, 200}, 0);
        }

        // Sample points along [a,b]
        for (int k = 0; k < samples_per_side; ++k) {
            const double t = (samples_per_side == 1) ? 0.5 : (static_cast<double>(k) / static_cast<double>(samples_per_side - 1));
            const double qx = static_cast<double>(a.x) + t * vx;
            const double qy = static_cast<double>(a.y) + t * vy;
            const double s_q = t * len;

            auto it = std::lower_bound(donors.begin(), donors.end(), s_q,
                                       [](const Donor& d, double v) { return d.s < v; });

            Donor d0 = donors.front();
            Donor d1 = donors.back();

            if (it == donors.begin()) {
                d0 = donors.front();
                d1 = donors.front();
            } else if (it == donors.end()) {
                d0 = donors.back();
                d1 = donors.back();
            } else {
                d1 = *it;
                d0 = *(it - 1);
            }

            const double denom = (d1.s - d0.s);
            const double w01 = (std::abs(denom) < 1e-9) ? 0.0 : std::clamp((s_q - d0.s) / denom, 0.0, 1.0);

            const point3u c0 = get_rgb(image_rgb_or_gray, d0.p);
            const point3u c1 = get_rgb(image_rgb_or_gray, d1.p);

            auto lerp_u8 = [&](std::uint8_t a0, std::uint8_t a1) -> std::uint8_t {
                const double v = (1.0 - w01) * static_cast<double>(a0) + w01 * static_cast<double>(a1);
                const int iv = static_cast<int>(std::lround(v));
                return static_cast<std::uint8_t>(std::clamp(iv, 0, 255));
            };

            colors.push_back(point3u{lerp_u8(c0.r, c1.r), lerp_u8(c0.g, c1.g), lerp_u8(c0.b, c1.b)});

            if (do_dbg) {
                // donors: blue
                put_point_rgb(dbg_side, d0.p.x, d0.p.y, point3u{0, 0, 255}, dbg->point_radius);
                put_point_rgb(dbg_side, d1.p.x, d1.p.y, point3u{0, 0, 255}, dbg->point_radius);

                // sample: red (rounded to nearest pixel for visualization)
                const int sx = static_cast<int>(std::lround(qx));
                const int sy = static_cast<int>(std::lround(qy));
                put_point_rgb(dbg_side, sx, sy, point3u{255, 0, 0}, dbg->point_radius);

                put_point_rgb(dbg_all, d0.p.x, d0.p.y, point3u{0, 0, 255}, dbg->point_radius);
                put_point_rgb(dbg_all, d1.p.x, d1.p.y, point3u{0, 0, 255}, dbg->point_radius);
                put_point_rgb(dbg_all, sx, sy, point3u{255, 0, 0}, dbg->point_radius);
            }
        }

        if (do_dbg) {
            std::ostringstream name;
            name << dbg->prefix << "side" << std::setw(2) << std::setfill('0') << si << "_sampling" << dbg->dump_ext;
            libimages::debug_io::dump_image((dbg->out_dir / name.str()).string(), dbg_side, dbg->verbose, dbg->force_overwrite);
        }

        out.push_back(std::move(colors));
    }

    if (do_dbg) {
        const std::string name = dbg->prefix + std::string("sampling_all") + dbg->dump_ext;
        libimages::debug_io::dump_image((dbg->out_dir / name).string(), dbg_all, dbg->verbose, dbg->force_overwrite);
    }

    return out;
}

libimages::image8u visualize_sides_overlay(const libimages::image8u& image_rgb_or_gray,
                                          const std::vector<std::vector<point2i>>& sides,
                                          std::uint32_t seed,
                                          std::uint8_t darken_div) {
    libimages::image8u out = to_rgb_darken(image_rgb_or_gray, darken_div);

    FastRandom rng(seed);
    for (const auto& side : sides) {
        const std::uint32_t rv = rng.nextU32();
        const point3u col{
            static_cast<std::uint8_t>(rv & 0xFFu),
            static_cast<std::uint8_t>((rv >> 8) & 0xFFu),
            static_cast<std::uint8_t>((rv >> 16) & 0xFFu),
        };
        for (const auto& p : side) {
            if (p.x < 0 || p.x >= out.width() || p.y < 0 || p.y >= out.height()) continue;
            out(p.y, p.x, 0) = col.r;
            out(p.y, p.x, 1) = col.g;
            out(p.y, p.x, 2) = col.b;
        }
    }
    return out;
}

libimages::image8u make_kxl_image(const std::vector<std::vector<point3u>>& samples) {
    const int K = static_cast<int>(samples.size());
    const int L = (K > 0) ? static_cast<int>(samples[0].size()) : 0;

    libimages::image8u out(L, K, 3);
    for (int j = 0; j < K; ++j) {
        rassert(static_cast<int>(samples[static_cast<std::size_t>(j)].size()) == L, "All sides must have same L");
        for (int i = 0; i < L; ++i) {
            const auto c = samples[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
            out(j, i, 0) = c.r;
            out(j, i, 1) = c.g;
            out(j, i, 2) = c.b;
        }
    }
    return out;
}

} // namespace find_mask_borders
