#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <vector>

namespace libimages {

template <typename T> class Image final {
  public:
    using value_type = T;

    Image();
    Image(int width, int height, int channels);

    int width() const noexcept;
    int height() const noexcept;
    int channels() const noexcept;

    // Number of elements in a single row (width * channels)
    std::size_t stride_elements() const noexcept;

    T *data() noexcept;
    const T *data() const noexcept;
    std::size_t size() const noexcept;

    void fill(const T &value);

    // Access for grayscale images (channels == 1)
    T &operator()(int j, int i, std::source_location loc = std::source_location::current());
    const T &operator()(int j, int i, std::source_location loc = std::source_location::current()) const;

    // Access for multi-channel images
    T &operator()(int j, int i, int c, std::source_location loc = std::source_location::current());
    const T &operator()(int j, int i, int c, std::source_location loc = std::source_location::current()) const;

  private:
    int w_ = 0;
    int h_ = 0;
    int c_ = 0;
    std::vector<T> data_;

    void check_bounds_2d(int j, int i, std::source_location loc) const;
    void check_bounds_3d(int j, int i, int c, std::source_location loc) const;
    std::size_t index(int j, int i, int c) const;
};

extern template class Image<std::uint8_t>;
extern template class Image<float>;

using image8u = Image<std::uint8_t>;
using image32i = Image<int>;
using image32f = Image<float>;

} // namespace libimages
