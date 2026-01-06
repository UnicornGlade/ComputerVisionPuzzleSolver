#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace libbase {

template <typename T> struct point2 final {
    T x{};
    T y{};

    point2() = default;
    point2(T x_, T y_);

    // Read/write access via operator[]
    T& operator[](std::size_t idx);
    const T& operator[](std::size_t idx) const;

    // Vector ops
    point2 operator+(const point2& rhs) const;
    point2 operator-(const point2& rhs) const;
    point2& operator+=(const point2& rhs);
    point2& operator-=(const point2& rhs);
    point2 operator-() const;

    // Scalar ops (same scalar type)
    point2 operator*(T s) const;
    point2 operator/(T s) const;
    point2& operator*=(T s);
    point2& operator/=(T s);

    // Dot product
    T dot(const point2& rhs) const;

    // Norm / length
    T norm2() const;
    double length() const;

    // String conversion
    std::string to_string() const;

    // Comparisons
    bool operator==(const point2& rhs) const;
    bool operator!=(const point2& rhs) const;
};

using point2f = point2<float>;
using point2i = point2<int>;

// Stream output
template <typename T> std::ostream& operator<<(std::ostream& os, const point2<T>& p);

// Scalar * vector (same scalar type)
point2i operator*(int s, const point2i& p);
point2f operator*(float s, const point2f& p);

// Mixed scalar for int-vectors (requested): point2i * float -> point2f, point2i / float -> point2f
point2f operator*(const point2i& p, float s);
point2f operator/(const point2i& p, float s);
point2f operator*(float s, const point2i& p);

// Prevent implicit template instantiation in every TU (definitions + explicit instantiation are in point2.cpp)
extern template struct point2<float>;
extern template struct point2<int>;
extern template std::ostream& operator<<(std::ostream& os, const point2<float>& p);
extern template std::ostream& operator<<(std::ostream& os, const point2<int>& p);

} // namespace libbase
