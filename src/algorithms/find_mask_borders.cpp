#include "find_mask_borders.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <libbase/algorithms/disjoint_set.h>
#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>

#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/sobel_gradients.h>
#include <libimages/debug_io.h>

namespace fs = std::filesystem;
using libimages::image8u;

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

static inline bool in_bounds(int x, int y, int w, int h) { return x >= 0 && x < w && y >= 0 && y < h; }

// signed area of cycle points in image coords (y down). clockwise => area < 0 typically.
static double signed_area_cycle(const std::vector<point2i>& P) {
    const int n = static_cast<int>(P.size());
    double a = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& p0 = P[i];
        const auto& p1 = P[(i + 1) % n];
        a += double(p0.x) * double(p1.y) - double(p1.x) * double(p0.y);
    }
    return 0.5 * a;
}

// Distance from point to segment using pixel centers.
static float dist_point_to_segment_center(point2i p, point2i a, point2i b) {
    const double px = double(p.x) + 0.5;
    const double py = double(p.y) + 0.5;
    const double ax = double(a.x) + 0.5;
    const double ay = double(a.y) + 0.5;
    const double bx = double(b.x) + 0.5;
    const double by = double(b.y) + 0.5;

    const double abx = bx - ax;
    const double aby = by - ay;
    const double apx = px - ax;
    const double apy = py - ay;

    const double denom = abx * abx + aby * aby;
    if (denom < 1e-18) {
        const double dx = px - ax;
        const double dy = py - ay;
        return float(std::sqrt(dx * dx + dy * dy));
    }

    double t = (apx * abx + apy * aby) / denom;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    const double cx = ax + t * abx;
    const double cy = ay + t * aby;

    const double dx = px - cx;
    const double dy = py - cy;
    return float(std::sqrt(dx * dx + dy * dy));
}

// Moore-neighbor tracing directions in CLOCKWISE order (image coords y down):
// E, SE, S, SW, W, NW, N, NE
static constexpr std::array<int, 8> kDx = { 1, 1, 0,-1,-1,-1, 0, 1 };
static constexpr std::array<int, 8> kDy = { 0, 1, 1, 1, 0,-1,-1,-1 };

static int dir_index_from_delta(int dx, int dy) {
    for (int i = 0; i < 8; ++i) {
        if (kDx[i] == dx && kDy[i] == dy) return i;
    }
    return -1;
}

// Find a boundary pixel of object (object==255, and has at least one 0 neighbor in 8-neighborhood, outside treated 0)
static bool find_object_boundary_start(const image8u& obj, point2i& out_start) {
    const int w = obj.width();
    const int h = obj.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (obj(y, x) != 255) continue;

            bool has_bg = false;
            for (int k = 0; k < 8; ++k) {
                const int xx = x + kDx[k];
                const int yy = y + kDy[k];
                if (!in_bounds(xx, yy, w, h)) { has_bg = true; break; }
                if (obj(yy, xx) == 0) { has_bg = true; break; }
            }
            if (has_bg) {
                out_start = point2i{x, y};
                return true;
            }
        }
    }
    return false;
}

// Robust contour extraction from object_mask (preferred).
// Returns a closed cycle of boundary pixels in clockwise order (if possible).
static std::vector<point2i> trace_boundary_moore(const image8u& object_mask) {
    check_mask01(object_mask);

    const int w = object_mask.width();
    const int h = object_mask.height();

    point2i start{-1, -1};
    if (!find_object_boundary_start(object_mask, start)) return {};

    // backtrack starts as W of start (background/outside)
    point2i b{start.x - 1, start.y};
    point2i c = start;

    std::vector<point2i> contour;
    contour.reserve(static_cast<std::size_t>(w + h) * 2);

    // Need second point to define stop condition.
    point2i second{-1, -1};
    int guard = 0;
    const int guard_max = w * h * 10;

    while (guard++ < guard_max) {
        contour.push_back(c);

        // direction from c to b (backtrack) must be one of 8 dirs if adjacent;
        // if not adjacent (at image boundary), clamp to W.
        int dbx = b.x - c.x;
        int dby = b.y - c.y;
        if (dbx < -1) dbx = -1; if (dbx > 1) dbx = 1;
        if (dby < -1) dby = -1; if (dby > 1) dby = 1;

        int idx_b = dir_index_from_delta(dbx, dby);
        if (idx_b < 0) idx_b = 4; // W

        // Search neighbors clockwise starting from next after backtrack.
        point2i next{-1, -1};
        point2i new_b = b;

        for (int s = 1; s <= 8; ++s) {
            const int idx = (idx_b + s) & 7;
            const int xx = c.x + kDx[idx];
            const int yy = c.y + kDy[idx];

            if (!in_bounds(xx, yy, w, h)) continue;
            if (object_mask(yy, xx) != 255) continue;

            next = point2i{xx, yy};

            // new backtrack is the neighbor just before idx in the scan order
            const int idx_prev = (idx - 1) & 7;
            new_b = point2i{c.x + kDx[idx_prev], c.y + kDy[idx_prev]};
            break;
        }

        if (next.x < 0) {
            // can't continue -> not a valid boundary
            return {};
        }

        if (second.x < 0) second = next;

        // termination: back to start and next is second
        if (c.x == start.x && c.y == start.y && next.x == second.x && next.y == second.y && contour.size() > 2) break;

        b = new_b;
        c = next;
    }

    if (contour.size() < 4) return {};

    // Ensure clockwise by area sign (clockwise in image coords tends to have area < 0)
    const double a = signed_area_cycle(contour);
    if (a > 0.0) std::reverse(contour.begin(), contour.end());

    return contour;
}

// Fallback: close 1-pixel gaps in border_mask (in-place) using 4 patterns.
static image8u fill_single_pixel_gaps(const image8u& border_mask) {
    check_mask01(border_mask);
    const int w = border_mask.width();
    const int h = border_mask.height();

    image8u out = border_mask;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            if (out(y, x) == 255) continue;

            const bool lr = (border_mask(y, x - 1) == 255) && (border_mask(y, x + 1) == 255);
            const bool ud = (border_mask(y - 1, x) == 255) && (border_mask(y + 1, x) == 255);
            const bool d1 = (border_mask(y - 1, x - 1) == 255) && (border_mask(y + 1, x + 1) == 255);
            const bool d2 = (border_mask(y - 1, x + 1) == 255) && (border_mask(y + 1, x - 1) == 255);

            if (lr || ud || d1 || d2) out(y, x) = 255;
        }
    }
    return out;
}

// Debug helper: show contour + corners + sides
static image8u visualize_contour_sides(const image8u& base_mask,
                                      const std::vector<point2i>& corners,
                                      const std::vector<std::vector<point2i>>* sides) {
    const int w = base_mask.width();
    const int h = base_mask.height();
    image8u rgb(w, h, 3);
    rgb.fill(std::uint8_t(0));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (base_mask(y, x) == 255) {
                rgb(y, x, 0) = 90;
                rgb(y, x, 1) = 90;
                rgb(y, x, 2) = 90;
            }
        }
    }

    auto put_disk = [&](int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b, int rad) {
        for (int dy = -rad; dy <= rad; ++dy)
            for (int dx = -rad; dx <= rad; ++dx) {
                const int xx = x + dx, yy = y + dy;
                if (!in_bounds(xx, yy, w, h)) continue;
                rgb(yy, xx, 0) = r; rgb(yy, xx, 1) = g; rgb(yy, xx, 2) = b;
            }
    };

    if (sides) {
        std::uint32_t seed = 123u;
        for (std::size_t si = 0; si < sides->size(); ++si) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            const std::uint8_t cr = std::uint8_t(seed & 0xFFu);
            const std::uint8_t cg = std::uint8_t((seed >> 8) & 0xFFu);
            const std::uint8_t cb = std::uint8_t((seed >> 16) & 0xFFu);

            for (const auto& p : (*sides)[si]) {
                rgb(p.y, p.x, 0) = cr;
                rgb(p.y, p.x, 1) = cg;
                rgb(p.y, p.x, 2) = cb;
            }
        }
    }

    for (const auto& c : corners) put_disk(c.x, c.y, 255, 0, 0, 2);
    return rgb;
}

static bool dbg_on(const SplitSidesDebugParams* dbg) {
    return dbg && !dbg->out_dir.empty();
}

static void dbg_dump(const SplitSidesDebugParams* dbg, const std::string& name, const image8u& img) {
    if (!dbg_on(dbg)) return;
    libimages::debug_io::dump_image((dbg->out_dir / (dbg->prefix + name + dbg->dump_ext)).string(),
                                    img,
                                    dbg->verbose,
                                    dbg->force_overwrite);
}

// ---------------------------------------------------------------
// NEW IMPLEMENTATION (cycle simplification until K segments)
// ---------------------------------------------------------------
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

    // 1) Get a robust ordered cycle.
    // Prefer object_mask contour tracing (handles degree>2 and border gaps).
    std::vector<point2i> P = trace_boundary_moore(object_mask);

    // Fallback: try to heal 1px gaps in border_mask and trace boundary of that as "object".
    if (P.empty()) {
        const image8u healed = fill_single_pixel_gaps(border_mask);
        P = trace_boundary_moore(healed);
        if (dbg_on(dbg)) dbg_dump(dbg, "cyc00_border_healed", healed);
    }

    if (P.size() < 4) return {};

    const int N = static_cast<int>(P.size());
    if (side_count >= N) {
        std::vector<std::vector<point2i>> out;
        out.reserve(static_cast<std::size_t>(N));
        for (const auto& p : P) out.push_back({p});
        return out;
    }

    // 2) Simplify the cycle by repeatedly removing the vertex with minimal local error.
    std::vector<int> prv(static_cast<std::size_t>(N));
    std::vector<int> nxt(static_cast<std::size_t>(N));
    std::vector<std::uint8_t> alive(static_cast<std::size_t>(N), 1);
    std::vector<std::uint32_t> ver(static_cast<std::size_t>(N), 0);

    for (int i = 0; i < N; ++i) {
        prv[std::size_t(i)] = (i - 1 + N) % N;
        nxt[std::size_t(i)] = (i + 1) % N;
    }

    auto calc_err = [&](int i) -> float {
        const int a = prv[std::size_t(i)];
        const int b = nxt[std::size_t(i)];
        if (a == b) return std::numeric_limits<float>::infinity();
        return dist_point_to_segment_center(P[std::size_t(i)], P[std::size_t(a)], P[std::size_t(b)]);
    };

    struct Node { float err; int i; std::uint32_t v; };
    struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.err > b.err; } };
    std::priority_queue<Node, std::vector<Node>, Cmp> pq;

    for (int i = 0; i < N; ++i) pq.push(Node{calc_err(i), i, ver[std::size_t(i)]});

    int alive_count = N;

    while (alive_count > side_count) {
        rassert(!pq.empty(), "split_border_into_sides: heap empty");

        const Node t = pq.top();
        pq.pop();

        if (t.i < 0 || t.i >= N) continue;
        if (!alive[std::size_t(t.i)]) continue;
        if (t.v != ver[std::size_t(t.i)]) continue;

        const int a = prv[std::size_t(t.i)];
        const int b = nxt[std::size_t(t.i)];
        if (a == b) break;

        // remove vertex t.i
        nxt[std::size_t(a)] = b;
        prv[std::size_t(b)] = a;
        alive[std::size_t(t.i)] = 0;
        --alive_count;

        // update neighbors
        for (int vtx : {a, b}) {
            if (!alive[std::size_t(vtx)]) continue;
            ++ver[std::size_t(vtx)];
            pq.push(Node{calc_err(vtx), vtx, ver[std::size_t(vtx)]});
        }
    }

    // 3) Collect remaining corners in cycle order (using nxt links).
    int start = -1;
    for (int i = 0; i < N; ++i) if (alive[std::size_t(i)]) { start = i; break; }
    rassert(start >= 0, "split_border_into_sides: no corners alive");

    std::vector<int> corners_idx;
    corners_idx.reserve(std::size_t(alive_count));
    {
        int cur = start;
        int guard = 0;
        do {
            rassert(guard++ < N + 10, "split_border_into_sides: corner traversal guard");
            corners_idx.push_back(cur);
            cur = nxt[std::size_t(cur)];
        } while (cur != start);
    }

    // 4) Build sides as arcs on ORIGINAL contour P between consecutive corners.
    std::vector<std::vector<point2i>> sides;
    sides.reserve(corners_idx.size());

    for (std::size_t si = 0; si < corners_idx.size(); ++si) {
        const int a = corners_idx[si];
        const int b = corners_idx[(si + 1) % corners_idx.size()];

        std::vector<point2i> side;
        int cur = a;
        int guard = 0;
        while (cur != b) {
            rassert(guard++ < N + 10, "split_border_into_sides: side guard");
            side.push_back(P[std::size_t(cur)]);
            cur = (cur + 1) % N;
        }
        sides.push_back(std::move(side));
    }

    // Debug dumps
    if (dbg_on(dbg)) {
        std::vector<point2i> corners_xy;
        corners_xy.reserve(corners_idx.size());
        for (int idx : corners_idx) corners_xy.push_back(P[std::size_t(idx)]);

        dbg_dump(dbg, "cyc01_contour_mask", border_mask);
        dbg_dump(dbg, "cyc02_corners", visualize_contour_sides(border_mask, corners_xy, nullptr));
        dbg_dump(dbg, "cyc03_sides", visualize_contour_sides(border_mask, corners_xy, &sides));
    }

    return sides;
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
