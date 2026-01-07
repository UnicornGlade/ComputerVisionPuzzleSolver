#include "utils.h"

#include <libimages/image.h>

#include <libbase/algorithms/disjoint_set.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

using libimages::image8u;

namespace utils {

namespace {

static void check_inputs_common(const image8u& mask, int w, int h) {
    rassert(mask.channels() == 1, "extract_objects_by_mask: mask must be 1-channel", mask.channels());
    rassert(mask.width() == w && mask.height() == h, "extract_objects_by_mask: mask size mismatch",
            mask.width(), mask.height(), w, h);

    // Optional strictness: ensure {0,255}.
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            const std::uint8_t v = mask(j, i);
            rassert(v == 0 || v == 255, "extract_objects_by_mask: mask must be {0,255}", int(v), j, i);
        }
}

static inline std::size_t idx_of(int i, int j, int w) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i);
}

} // namespace

template <typename T>
std::vector<ExtractedObject<T>> extract_objects_by_mask(const libimages::Image<T>& img,
                                                        const libimages::image8u& mask,
                                                        const ExtractParams& p) {
    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();

    check_inputs_common(mask, w, h);
    rassert(c > 0, "extract_objects_by_mask: invalid channels", c);

    const std::size_t N = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    DisjointSetUnion dsu(N);

    // Union pass
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (mask(j, i) != 255) continue;

            const std::size_t id = idx_of(i, j, w);

            // Right
            if (i + 1 < w && mask(j, i + 1) == 255) dsu.unite(id, idx_of(i + 1, j, w));
            // Down
            if (j + 1 < h && mask(j + 1, i) == 255) dsu.unite(id, idx_of(i, j + 1, w));

            if (p.eight_connected) {
                // Down-right
                if (i + 1 < w && j + 1 < h && mask(j + 1, i + 1) == 255)
                    dsu.unite(id, idx_of(i + 1, j + 1, w));
                // Down-left
                if (i - 1 >= 0 && j + 1 < h && mask(j + 1, i - 1) == 255)
                    dsu.unite(id, idx_of(i - 1, j + 1, w));
            }
        }
    }

    // Accumulate bbox and counts by root
    std::vector<bbox2i> bb(N);
    std::vector<int> cnt(N, 0);

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (mask(j, i) != 255) continue;
            const std::size_t id = idx_of(i, j, w);
            const std::size_t r = dsu.find(id);
            bb[r].include_pixel(i, j);
            cnt[r] += 1;
        }
    }

    // Collect roots
    struct RootInfo {
        std::size_t root = 0;
        bbox2i bbox;
        point2i offset;
    };

    std::vector<RootInfo> roots;
    roots.reserve(128);

    for (std::size_t r = 0; r < N; ++r) {
        if (cnt[r] <= 0) continue;
        rassert(!bb[r].is_empty(), "internal: bbox empty but cnt>0", r, cnt[r]);
        roots.push_back(RootInfo{r, bb[r], point2i{bb[r].min.x, bb[r].min.y}});
    }

    // Deterministic order: sort by (y,x)
    std::sort(roots.begin(), roots.end(), [](const RootInfo& a, const RootInfo& b) {
        if (a.offset.y != b.offset.y) return a.offset.y < b.offset.y;
        return a.offset.x < b.offset.x;
    });

    // Map root -> out index
    std::unordered_map<std::size_t, int> root_to_idx;
    root_to_idx.reserve(roots.size() * 2 + 1);
    for (int k = 0; k < static_cast<int>(roots.size()); ++k) root_to_idx[roots[static_cast<std::size_t>(k)].root] = k;

    // Allocate outputs
    std::vector<ExtractedObject<T>> out;
    out.resize(roots.size());

    for (std::size_t k = 0; k < roots.size(); ++k) {
        const auto& ri = roots[k];
        const int ww = ri.bbox.width();
        const int hh = ri.bbox.height();

        out[k].bbox = ri.bbox;
        out[k].offset = ri.offset;
        out[k].image = libimages::Image<T>(ww, hh, c);
        out[k].mask = libimages::image8u(ww, hh, 1);

        // Initialize to 0 (holes stay black)
        for (std::size_t t = 0; t < out[k].image.size(); ++t) out[k].image.data()[t] = T(0);
        out[k].mask.fill(static_cast<std::uint8_t>(0));
    }

    // Fill outputs
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            if (mask(j, i) != 255) continue;

            const std::size_t id = idx_of(i, j, w);
            const std::size_t r = dsu.find(id);

            const auto it = root_to_idx.find(r);
            if (it == root_to_idx.end()) continue;
            const int oi = it->second;

            const int x0 = out[static_cast<std::size_t>(oi)].offset.x;
            const int y0 = out[static_cast<std::size_t>(oi)].offset.y;
            const int ii = i - x0;
            const int jj = j - y0;

            out[static_cast<std::size_t>(oi)].mask(jj, ii) = 255;
            for (int cc = 0; cc < c; ++cc) {
                out[static_cast<std::size_t>(oi)].image(jj, ii, cc) = img(j, i, cc);
            }
        }
    }

    return out;
}

// Explicit instantiations
template std::vector<ExtractedObject<std::uint8_t>>
extract_objects_by_mask<std::uint8_t>(const libimages::Image<std::uint8_t>&,
                                      const libimages::image8u&,
                                      const ExtractParams&);

template std::vector<ExtractedObject<float>>
extract_objects_by_mask<float>(const libimages::Image<float>&,
                               const libimages::image8u&,
                               const ExtractParams&);

} // namespace libimages::utils
