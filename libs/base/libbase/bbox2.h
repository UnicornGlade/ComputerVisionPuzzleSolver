#pragma once

#include <algorithm>
#include <type_traits>

#include <libbase/point2.h>
#include <libbase/runtime_assert.h>

// Axis-aligned bounding box in 2D.
// For pixel coordinates we use half-open convention: [min, max), where max is exclusive.
template <typename T> struct bbox2 final {
    point2<T> min{};
    point2<T> max{};
    bool empty = true;

    bbox2() = default;

    static bbox2 make_empty() { return bbox2{}; }

    bool is_empty() const noexcept { return empty; }

    T width() const noexcept { return empty ? T(0) : (max.x - min.x); }
    T height() const noexcept { return empty ? T(0) : (max.y - min.y); }

    point2<T> size() const noexcept { return point2<T>{width(), height()}; }

    // For continuous boxes (or if you want inclusive bounds semantics manually).
    void include_point(const point2<T>& p) {
        if (empty) {
            min = p;
            max = p;
            empty = false;
            return;
        }
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
    }

    void include_box(const bbox2& b) {
        if (b.empty) return;
        if (empty) {
            *this = b;
            return;
        }
        min.x = std::min(min.x, b.min.x);
        min.y = std::min(min.y, b.min.y);
        max.x = std::max(max.x, b.max.x);
        max.y = std::max(max.y, b.max.y);
    }

    // Pixel bbox helper: includes a single pixel at integer coordinates (x,y).
    template <typename U = T, typename = std::enable_if_t<std::is_same_v<U, int>>>
    void include_pixel(int x, int y) {
        if (empty) {
            min = point2i{x, y};
            max = point2i{x + 1, y + 1};
            empty = false;
            return;
        }
        min.x = std::min(min.x, x);
        min.y = std::min(min.y, y);
        max.x = std::max(max.x, x + 1);
        max.y = std::max(max.y, y + 1);
    }

    template <typename U = T, typename = std::enable_if_t<std::is_same_v<U, int>>>
    bool contains_pixel(int x, int y) const noexcept {
        if (empty) return false;
        return x >= min.x && x < max.x && y >= min.y && y < max.y;
    }
};

using bbox2f = bbox2<float>;
using bbox2i = bbox2<int>;

// Avoid implicit template instantiation in every TU
extern template struct bbox2<float>;
extern template struct bbox2<int>;
