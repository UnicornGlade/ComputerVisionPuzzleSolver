#include "downscale.h"

#include <algorithm>
#include <cmath>

#include <libbase/runtime_assert.h>

namespace libimages {

float normalize_downscale_ratio(float r) {
    rassert(r > 0.0f, "kDownscaleRatio must be > 0", r);
    if (r > 1.0f) r = 1.0f / r;
    rassert(r > 0.0f && r <= 1.0f, "normalized ratio must be in (0,1]", r);
    return r;
}

image8u downscale_nearest(const image8u& img, float downscale_ratio) {
    const float s = normalize_downscale_ratio(downscale_ratio);

    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    rassert(w > 0 && h > 0 && c > 0, "Invalid input image", w, h, c);

    if (std::fabs(s - 1.0f) < 1e-7f) return img;

    const int nw = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * s)));
    const int nh = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * s)));

    image8u out(nw, nh, c);

    const float inv_s = 1.0f / s;
    for (int j = 0; j < nh; ++j) {
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

} // namespace libimages
