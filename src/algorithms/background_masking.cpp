#include "background_masking.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <libbase/runtime_assert.h>
#include <libbase/stats.h>
#include <libimages/algorithms/morphology.h>

namespace background_masking {

static void check_grayscale_u8(const libimages::image8u& img) {
    rassert(img.channels() == 1, "background_masking expects 1-channel image", img.channels());
    rassert(img.width() > 0 && img.height() > 0, "background_masking expects non-empty image", img.width(), img.height());
}

static std::vector<std::uint8_t> collect_perimeter_u8(const libimages::image8u& gray) {
    const int w = gray.width();
    const int h = gray.height();

    std::vector<std::uint8_t> vals;
    vals.reserve(static_cast<std::size_t>(2 * w + 2 * h));

    // Top row
    for (int i = 0; i < w; ++i) vals.push_back(gray(0, i));

    // Bottom row (if different)
    if (h > 1) {
        for (int i = 0; i < w; ++i) vals.push_back(gray(h - 1, i));
    }

    // Left/right columns excluding corners (to avoid duplicates)
    for (int j = 1; j < h - 1; ++j) {
        vals.push_back(gray(j, 0));
        if (w > 1) vals.push_back(gray(j, w - 1));
    }

    return vals;
}

float estimate_background_threshold(const libimages::image8u& gray) {
    check_grayscale_u8(gray);

    const auto per = collect_perimeter_u8(gray);
    rassert(!per.empty(), "Perimeter is empty (unexpected)", gray.width(), gray.height());

    const double p90 = stats::percentile(per, 90.0);

    std::cout << "[background_masking] perimeter stats: " << stats::summaryStats(per) << "\n";
    std::cout << "[background_masking] perimeter p90=" << p90 << " -> threshold=1.5*p90\n";

    const double thr = 1.5 * p90;
    const double thr_clamped = std::clamp(thr, 0.0, 255.0);
    return static_cast<float>(thr_clamped);
}

libimages::image8u build_foreground_mask(const libimages::image8u& gray, const Params& p) {
    check_grayscale_u8(gray);
    rassert(p.dilate_strength >= 0, "dilate_strength must be >= 0", p.dilate_strength);
    rassert(p.erode_strength >= 0, "erode_strength must be >= 0", p.erode_strength);

    const float thr = estimate_background_threshold(gray);

    libimages::image8u mask(gray.width(), gray.height(), 1);
    for (int j = 0; j < gray.height(); ++j) {
        for (int i = 0; i < gray.width(); ++i) {
            const std::uint8_t v = gray(j, i);
            mask(j, i) = (static_cast<float>(v) > thr) ? 255 : 0;
        }
    }

    // Refine using morphology: closing (dilate then erode)
    if (p.dilate_strength > 0) mask = morphology::dilate(mask, p.dilate_strength);
    if (p.erode_strength > 0) mask = morphology::erode(mask, p.erode_strength);

    return mask;
}

} // namespace background_masking
