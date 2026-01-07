#include "find_mask_borders.h"

#include "libimages/algorithms/gaussian_blur.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

#include <libbase/algorithms/disjoint_set.h>
#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

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

static void draw_line_rgb(libimages::image8u& img, point2i a, point2i b, point3u c, int rad) {
    int x0 = a.x, y0 = a.y;
    const int x1 = b.x, y1 = b.y;
    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        put_point_rgb(img, x0, y0, c, rad);
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
                                                         int side_count) {
    rassert(side_count >= 1, "side_count must be >= 1", side_count);
    check_mask01(object_mask);
    check_mask01(border_mask);
    rassert(object_mask.width() == border_mask.width() && object_mask.height() == border_mask.height(),
            "object_mask and border_mask must have same size",
            object_mask.width(), object_mask.height(), border_mask.width(), border_mask.height());

    const int w = object_mask.width();
    const int h = object_mask.height();

    // --- NEW: padding for better Sobel stability near borders ---
    constexpr int pad = 10;
    const int wp = w + 2 * pad;
    const int hp = h + 2 * pad;

    // Convert object_mask to float image for sobel, but in a padded buffer.
    libimages::image32f m(wp, hp, 1);
    m.fill(0.0f);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            m(j + pad, i + pad) = static_cast<float>(object_mask(j, i));
        }
    }

    // Blur before Sobel (as you already did), but now on padded image.
    m = libimages::gaussian_blur_gray(m, 0.5f);

    const libimages::Gradients g = libimages::sobel_gradients(m);

    // Helper to read padded gradients using original coordinates.
    auto angle_at = [&](int y, int x) -> float {
        return g.angle(y + pad, x + pad);
    };

    // Collect border pixel ids + mapping to compact indices.
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

        // --- CHANGED: read angles from padded gradients ---
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

    std::sort(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) { return e1.ang_diff < e2.ang_diff; });

    DisjointSetUnion dsu(border_ids.size());
    int comps = static_cast<int>(border_ids.size());

    for (const auto& e : edges) {
        if (comps <= side_count) break;
        const int ia = id_to_idx[static_cast<std::size_t>(e.a)];
        const int ib = id_to_idx[static_cast<std::size_t>(e.b)];
        if (ia < 0 || ib < 0) continue;
        if (dsu.unite(static_cast<std::size_t>(ia), static_cast<std::size_t>(ib))) comps -= 1;
    }

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

    struct SideMeta {
        std::vector<point2i> pix;
        double ang = 0.0;
    };

    std::vector<SideMeta> meta;
    meta.reserve(groups.size());

    for (auto& s : groups) {
        double sx = 0.0, sy = 0.0;
        for (const auto& p : s) {
            sx += p.x;
            sy += p.y;
        }
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
        if (s.size() <= 1) {
            out.push_back(std::move(s));
            continue;
        }

        const float mean_inward_deg = mean_inward_deg_for_side(s);
        const double a = static_cast<double>(mean_inward_deg) * (3.14159265358979323846 / 180.0);
        const double nx = std::cos(a);
        const double ny = std::sin(a);

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
