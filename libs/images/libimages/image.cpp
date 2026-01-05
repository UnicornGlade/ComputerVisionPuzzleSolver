#include <libimages/image.h>

#include "libbase/runtime_assert.h"

#include <algorithm>
#include <string>

namespace libimages {

template <typename T> Image<T>::Image() = default;

template <typename T>
Image<T>::Image(int width, int height, int channels)
    : w_(width), h_(height), c_(channels),
      data_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels)) {
    rassert(width > 0 && height > 0 && channels > 0, "Invalid image size", width, height, channels);
}

template <typename T> int Image<T>::width() const noexcept { return w_; }

template <typename T> int Image<T>::height() const noexcept { return h_; }

template <typename T> int Image<T>::channels() const noexcept { return c_; }

template <typename T> std::size_t Image<T>::stride_elements() const noexcept {
    return static_cast<std::size_t>(w_) * static_cast<std::size_t>(c_);
}

template <typename T> T *Image<T>::data() noexcept { return data_.data(); }

template <typename T> const T *Image<T>::data() const noexcept { return data_.data(); }

template <typename T> std::size_t Image<T>::size() const noexcept { return data_.size(); }

template <typename T> void Image<T>::fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }

template <typename T> void Image<T>::check_bounds_2d(int j, int i, std::source_location loc) const {
    rassert(i >= 0 && i < w_ && j >= 0 && j < h_, 78497218931,
            "Pixel out of bounds:", "row j=" + std::to_string(j) + "/height=" + std::to_string(h_) + ",",
            "column i=" + std::to_string(i) + "/width=" + std::to_string(w_), format_code_location(loc));
}

template <typename T> void Image<T>::check_bounds_3d(int j, int i, int c, std::source_location loc) const {
    check_bounds_2d(j, i, loc);

    rassert(c >= 0 && c < c_, 65735424321,
            "Channel out of bounds:", "c=" + std::to_string(c) + "/channels count=" + std::to_string(c_),
            format_code_location(loc));
}

template <typename T> std::size_t Image<T>::index(int j, int i, int c) const {
    return (static_cast<std::size_t>(j) * static_cast<std::size_t>(w_) + static_cast<std::size_t>(i)) *
               static_cast<std::size_t>(c_) +
           static_cast<std::size_t>(c);
}

template <typename T> T &Image<T>::operator()(int j, int i, std::source_location loc) {
    rassert(c_ == 1, "(j,i) access is only valid for grayscale images", c_);
    check_bounds_2d(j, i, loc);
    return data_[index(j, i, 0)];
}

template <typename T> const T &Image<T>::operator()(int j, int i, std::source_location loc) const {
    rassert(c_ == 1, "(j,i) access is only valid for grayscale images", c_);
    check_bounds_2d(j, i, loc);
    return data_[index(j, i, 0)];
}

template <typename T> T &Image<T>::operator()(int j, int i, int c, std::source_location loc) {
    check_bounds_3d(j, i, c, loc);
    return data_[index(j, i, c)];
}

template <typename T> const T &Image<T>::operator()(int j, int i, int c, std::source_location loc) const {
    check_bounds_3d(j, i, c, loc);
    return data_[index(j, i, c)];
}

// Explicit instantiations (avoid recompiling template code in every TU)
template class Image<std::uint8_t>;
template class Image<float>;

} // namespace libimages
