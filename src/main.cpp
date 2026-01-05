#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>
#include <libimages/algorithms/downscale.h>
#include <libimages/algorithms/gaussian_blur.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/algorithms/sobel_gradients.h>

#include "disjoint_set.h"

using namespace libimages;

// All tunable parameters live here.
namespace cfg {
// Output naming.
inline constexpr const char *kDefaultDumpExt = ".jpg"; // png is lossless, but jpg is faster

// Debug IO behavior.
inline constexpr bool kDumpVerbose = true;        // true -> print extra info while saving.
inline constexpr bool kDumpForceOverwrite = true; // true -> overwrite existing files.

// Preprocessing.
inline constexpr float kDownscaleRatio = 32.0f; // 1.0 -> no resize. <1 shrinks. >1 is treated as 1/kDownscaleRatio.
inline constexpr float kGaussianSigma =
    0.01f; // Larger -> more smoothing (less noise, fewer edges). Smaller -> sharper but noisier.

// Graph-based gradient clustering (Olson 2011, Eq.(1)).
inline constexpr float kClusterKD =
    1000.0f; // Larger -> allow more angular spread, merges more components. Smaller -> splits more.
inline constexpr float kClusterKM =
    20000.0f; // Larger -> allow more magnitude spread, merges more. Smaller -> splits more.

}

// NOTE: All comments are in English by request.
namespace fs = std::filesystem;

namespace {

using libimages::image32f;
using libimages::image8u;

static std::string stage_path(const fs::path& out_dir, int step, const std::string& name,
                              const std::string& ext = cfg::kDefaultDumpExt) {
  const std::string filename = (step < 10 ? "0" : "") + std::to_string(step) + "_" + name + ext;
  return (out_dir / filename).string();
}

static std::string iters_dir_name(std::size_t unions_done) {
  // iters000000010 (9 digits)
  std::ostringstream oss;
  oss << "iters" << std::setw(9) << std::setfill('0') << unions_done;
  return oss.str();
}

struct CompStats {
  // Stored for each DSU root index.
  std::vector<float> min_mag;
  std::vector<float> max_mag;
  std::vector<float> min_ang; // degrees, unwrapped
  std::vector<float> max_ang; // degrees, unwrapped
};

static float center(float mn, float mx) { return 0.5f * (mn + mx); }

// Align component B to A by shifting B by multiples of 360 degrees.
static float compute_shift_360(float centerA, float centerB) {
  const float k = std::round((centerA - centerB) / 360.0f);
  return k * 360.0f;
}

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

static void merge_stats_after_union(const DisjointSetUnion& dsu, CompStats& s,
                                   std::size_t root_kept, std::size_t root_absorbed) {
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

  (void)dsu;
}

// ---- Snapshot visualizations ----

static image32f make_image_component_size_log(const DisjointSetUnion& dsu, int w, int h) {
  image32f out(w, h, 1);
  out.fill(0.0f);

  std::size_t max_sz = 1;
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      max_sz = std::max(max_sz, dsu.set_size(static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i));

  const float max_log = std::log2(static_cast<float>(max_sz));

  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
      const std::size_t sz = dsu.set_size(idx);
      if (sz <= 1) {
        out(j, i) = 0.0f;
        continue;
      }
      const float v = std::log2(static_cast<float>(sz)) / (max_log > 1e-6f ? max_log : 1.0f);
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

// Hue = component mean direction, Value = 1, background black for singletons.
static image8u make_image_direction_mean_hsv(const DisjointSetUnion& dsu, const CompStats& s, int w, int h) {
  image8u out(w, h, 3);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      out(j, i, 0) = 0;
      out(j, i, 1) = 0;
      out(j, i, 2) = 0;
    }
  }

  // Small HSV->RGB helper (copied from gradients.cpp logic style, but local to keep modules minimal).
  auto hsv_to_rgb = [](float h01, float s01, float v01, std::uint8_t* r, std::uint8_t* g, std::uint8_t* b) {
    const float hh = h01 * 6.0f;
    const int ii = static_cast<int>(std::floor(hh));
    const float f = hh - static_cast<float>(ii);

    const float p = v01 * (1.0f - s01);
    const float q = v01 * (1.0f - s01 * f);
    const float t = v01 * (1.0f - s01 * (1.0f - f));

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    switch (ii % 6) {
      case 0: rr = v01; gg = t;  bb = p;  break;
      case 1: rr = q;  gg = v01; bb = p;  break;
      case 2: rr = p;  gg = v01; bb = t;  break;
      case 3: rr = p;  gg = q;  bb = v01; break;
      case 4: rr = t;  gg = p;  bb = v01; break;
      case 5: rr = v01; gg = p;  bb = q;  break;
    }
    auto to_u8 = [](float x) -> std::uint8_t {
      int u = static_cast<int>(std::lround(x * 255.0f));
      if (u < 0) u = 0;
      if (u > 255) u = 255;
      return static_cast<std::uint8_t>(u);
    };
    *r = to_u8(rr);
    *g = to_u8(gg);
    *b = to_u8(bb);
  };

  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
      if (dsu.set_size(idx) <= 1) continue;

      const std::size_t r = dsu.find(idx);
      const float mean_unwrapped = center(s.min_ang[r], s.max_ang[r]);
      const float mean = wrap_angle_deg(mean_unwrapped);
      const float hue = mean / 360.0f;

      std::uint8_t rr, gg, bb;
      hsv_to_rgb(hue, 1.0f, 1.0f, &rr, &gg, &bb);
      out(j, i, 0) = rr;
      out(j, i, 1) = gg;
      out(j, i, 2) = bb;
    }
  }

  return out;
}

static void dump_unionfind_snapshot(const fs::path& out_root, std::size_t unions_done,
                                   DisjointSetUnion& dsu, const CompStats& stats,
                                   int w, int h) {
  const fs::path dir = out_root / iters_dir_name(unions_done);
  fs::create_directories(dir);

  // 1) component sizes (log-scaled)
  libimages::debug_io::dump_image((dir / ("00_component_sizes_log" + std::string(cfg::kDefaultDumpExt))).string(),
                                  make_image_component_size_log(dsu, w, h),
                                  cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

  // 2) mean direction (HSV)
  libimages::debug_io::dump_image((dir / ("01_direction_mean_hsv" + std::string(cfg::kDefaultDumpExt))).string(),
                                  make_image_direction_mean_hsv(dsu, stats, w, h),
                                  cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

  // 3) direction range
  libimages::debug_io::dump_image((dir / ("02_direction_range" + std::string(cfg::kDefaultDumpExt))).string(),
                                  make_image_direction_range(dsu, stats, w, h),
                                  cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

  // 4) magnitude range
  libimages::debug_io::dump_image((dir / ("03_magnitude_range" + std::string(cfg::kDefaultDumpExt))).string(),
                                  make_image_magnitude_range(dsu, stats, w, h),
                                  cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
}

struct Edge {
  std::size_t a = 0;
  std::size_t b = 0;
  float w = 0.0f; // angle diff
};

static std::vector<std::size_t> make_snapshot_schedule(std::size_t max_unions_possible) {
  std::vector<std::size_t> t;
  if (max_unions_possible >= 10) t.push_back(10);

  for (std::size_t x = 100; x <= 1000 && x <= max_unions_possible; x += 100) t.push_back(x);

  for (std::size_t x = 10000; x <= max_unions_possible; x *= 10) t.push_back(x);

  // Ensure strictly increasing.
  t.erase(std::unique(t.begin(), t.end()), t.end());
  return t;
}

} // namespace

int main(int argc, char** argv) {
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
      if (out_dir.empty()) out_dir = ".";
    }
    fs::create_directories(out_dir);

    // 00: input
    image8u input = libimages::load_image(input_path.string());
    libimages::debug_io::dump_image(stage_path(out_dir, 0, "input_image"), input,
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

    // 01: downscale
    input = downscale_nearest(input, cfg::kDownscaleRatio);
    libimages::debug_io::dump_image(stage_path(out_dir, 1, "input_downscaled"), input,
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

    // 02: grayscale
    const image32f gray = to_grayscale_float(input);
    libimages::debug_io::dump_image(stage_path(out_dir, 2, "gray"), gray,
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

    // 03: blur
    const image32f blurred = gaussian_blur_gray(gray, cfg::kGaussianSigma);
    libimages::debug_io::dump_image(stage_path(out_dir, 3, "gray_blurred"), blurred,
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

    // 04..: gradients
    const Gradients gr = sobel_gradients(blurred);
    libimages::debug_io::dump_image(stage_path(out_dir, 4, "sobel_dx_signed"),
                                    visualize_signed_to_u8(gr.dx),
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
    libimages::debug_io::dump_image(stage_path(out_dir, 5, "sobel_dy_signed"),
                                    visualize_signed_to_u8(gr.dy),
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
    libimages::debug_io::dump_image(stage_path(out_dir, 6, "sobel_mag"),
                                    gr.mag,
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);
    libimages::debug_io::dump_image(stage_path(out_dir, 7, "sobel_angle_hsv"),
                                    visualize_angle_hsv(gr.angle, gr.mag),
                                    cfg::kDumpVerbose, cfg::kDumpForceOverwrite);

    // ---- Build 8-connected edges ----
    const int w = gr.mag.width();
    const int h = gr.mag.height();
    rassert(gr.angle.width() == w && gr.angle.height() == h, "mag/angle size mismatch");

    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    DisjointSetUnion dsu(n);

    CompStats stats;
    stats.min_mag.resize(n);
    stats.max_mag.resize(n);
    stats.min_ang.resize(n);
    stats.max_ang.resize(n);

    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        const std::size_t idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;
        const float a = wrap_angle_deg(gr.angle(j, i));
        const float m = gr.mag(j, i);
        stats.min_ang[idx] = a;
        stats.max_ang[idx] = a;
        stats.min_mag[idx] = m;
        stats.max_mag[idx] = m;
      }
    }

    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>((w - 1) * h) +
                  static_cast<std::size_t>(w * (h - 1)) +
                  static_cast<std::size_t>(2 * (w - 1) * (h - 1)));

    for (int j = 1; j < h - 1; ++j) {
      for (int i = 1; i < w - 1; ++i) {
        const std::size_t a = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i;

        if (i + 1 < w - 1) {
          const std::size_t b = static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + (i + 1);
          edges.push_back(Edge{a, b, angle_diff_deg(gr.angle(j, i), gr.angle(j, i + 1))});
        }
        if (j + 1 < h - 1) {
          const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + i;
          edges.push_back(Edge{a, b, angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i))});
        }
        if (i + 1 < w - 1 && j + 1 < h - 1) {
          const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + (i + 1);
          edges.push_back(Edge{a, b, angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i + 1))});
        }
        if (i - 1 >= 1 && j + 1 < h - 1) {
          const std::size_t b = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(w) + (i - 1);
          edges.push_back(Edge{a, b, angle_diff_deg(gr.angle(j, i), gr.angle(j + 1, i - 1))});
        }
      }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) { return e1.w < e2.w; });

    // ---- Union process with snapshots ----
    const std::size_t max_unions_possible = (n > 0 ? n - 1 : 0);
    const std::vector<std::size_t> schedule = make_snapshot_schedule(max_unions_possible);
    std::size_t sched_pos = 0;

    std::size_t unions_done = 0;

    for (const Edge& e : edges) {
      const std::size_t ra = dsu.find(e.a);
      const std::size_t rb = dsu.find(e.b);
      if (ra == rb) continue;

      if (!can_merge_olson(dsu, stats, ra, rb, cfg::kClusterKD, cfg::kClusterKM)) continue;

      const auto [kept, absorbed] = dsu.unite_roots(ra, rb);
      if (kept == absorbed) continue;

      merge_stats_after_union(dsu, stats, kept, absorbed);
      ++unions_done;

      // Snapshot when reaching scheduled thresholds (after N successful unions).
      while (sched_pos < schedule.size() && unions_done == schedule[sched_pos]) {
        dump_unionfind_snapshot(out_dir, unions_done, dsu, stats, w, h);
        ++sched_pos;
      }

      if (sched_pos >= schedule.size()) {
        // We already dumped all requested snapshots.
        // Continue merging if you still want final segmentation later, but user asked snapshots only.
        // We keep going to be consistent with clustering.
      }
    }

    std::cout << "Saved debug images to: " << out_dir.string() << "\n";
    std::cout << "Total successful unions: " << unions_done << "\n";
    std::cout << "Edges processed: " << edges.size() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
