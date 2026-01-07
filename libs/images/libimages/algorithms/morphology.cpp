#include "morphology.h"

#include <algorithm>

#include <libbase/runtime_assert.h>

namespace morphology {

static void check_binary_01_255(const libimages::image8u& src) {
    rassert(src.channels() == 1, "morphology expects 1-channel image", src.channels());
    for (int j = 0; j < src.height(); ++j) {
        for (int i = 0; i < src.width(); ++i) {
            const std::uint8_t v = src(j, i);
            rassert(v == 0 || v == 255, "morphology expects binary pixels {0,255}", int(v), j, i);
        }
    }
}

libimages::image8u erode(const libimages::image8u& src, int strength) {
    rassert(strength >= 0, "erode: strength must be >= 0", strength);
    check_binary_01_255(src);

    const int w = src.width();
    const int h = src.height();

    libimages::image8u dst(w, h, 1);

    if (strength == 0) {
        dst = src;
        return dst;
    }

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            // Zero padding: if the neighborhood goes outside, erosion must be 0.
            if (j - strength < 0 || j + strength >= h || i - strength < 0 || i + strength >= w) {
                dst(j, i) = 0;
                continue;
            }

            bool all_on = true;
            for (int y = j - strength; y <= j + strength && all_on; ++y) {
                for (int x = i - strength; x <= i + strength; ++x) {
                    if (src(y, x) == 0) {
                        all_on = false;
                        break;
                    }
                }
            }
            dst(j, i) = all_on ? 255 : 0;
        }
    }

    return dst;
}

libimages::image8u dilate(const libimages::image8u& src, int strength) {
    rassert(strength >= 0, "dilate: strength must be >= 0", strength);
    check_binary_01_255(src);

    const int w = src.width();
    const int h = src.height();

    libimages::image8u dst(w, h, 1);

    if (strength == 0) {
        dst = src;
        return dst;
    }

    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int y0 = std::max(0, j - strength);
            const int y1 = std::min(h - 1, j + strength);
            const int x0 = std::max(0, i - strength);
            const int x1 = std::min(w - 1, i + strength);

            bool any_on = false;
            for (int y = y0; y <= y1 && !any_on; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (src(y, x) == 255) {
                        any_on = true;
                        break;
                    }
                }
            }
            dst(j, i) = any_on ? 255 : 0;
        }
    }

    return dst;
}

} // namespace morphology
