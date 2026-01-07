#include "point2.h"

#include <cmath>
#include <limits>
#include <ostream>
#include <sstream>
#include <type_traits>

#include <libbase/runtime_assert.h>

template <typename T> point2<T>::point2(T x_, T y_) : x(x_), y(y_) {}

template <typename T> T& point2<T>::operator[](std::size_t idx) {
  rassert(idx < 2, "point2 index out of bounds", idx);
  return (idx == 0) ? x : y;
}

template <typename T> const T& point2<T>::operator[](std::size_t idx) const {
  rassert(idx < 2, "point2 index out of bounds", idx);
  return (idx == 0) ? x : y;
}

template <typename T> point2<T> point2<T>::operator+(const point2& rhs) const { return point2(x + rhs.x, y + rhs.y); }

template <typename T> point2<T> point2<T>::operator-(const point2& rhs) const { return point2(x - rhs.x, y - rhs.y); }

template <typename T> point2<T>& point2<T>::operator+=(const point2& rhs) {
  x += rhs.x;
  y += rhs.y;
  return *this;
}

template <typename T> point2<T>& point2<T>::operator-=(const point2& rhs) {
  x -= rhs.x;
  y -= rhs.y;
  return *this;
}

template <typename T> point2<T> point2<T>::operator-() const { return point2(-x, -y); }

template <typename T> point2<T> point2<T>::operator*(T s) const { return point2(x * s, y * s); }

template <typename T> point2<T> point2<T>::operator/(T s) const {
  if constexpr (std::is_integral_v<T>) {
    rassert(s != 0, "division by zero");
    return point2(x / s, y / s); // integer division (truncate)
  } else {
    rassert(std::fabs(s) > 0.0, "division by zero");
    return point2(x / s, y / s);
  }
}

template <typename T> point2<T>& point2<T>::operator*=(T s) {
  x *= s;
  y *= s;
  return *this;
}

template <typename T> point2<T>& point2<T>::operator/=(T s) {
  *this = *this / s;
  return *this;
}

template <typename T> static void rassert_int_fits(long long v, const char* what) {
  const long long mn = static_cast<long long>(std::numeric_limits<int>::min());
  const long long mx = static_cast<long long>(std::numeric_limits<int>::max());
  rassert(v >= mn && v <= mx, "int overflow", what, v);
}

template <typename T> T point2<T>::dot(const point2& rhs) const {
  if constexpr (std::is_same_v<T, int>) {
    const long long a = static_cast<long long>(x) * static_cast<long long>(rhs.x);
    const long long b = static_cast<long long>(y) * static_cast<long long>(rhs.y);
    const long long s = a + b;

    rassert_int_fits<int>(a, "dot mul x");
    rassert_int_fits<int>(b, "dot mul y");
    rassert_int_fits<int>(s, "dot sum");

    return static_cast<int>(s);
  } else {
    return static_cast<T>(x * rhs.x + y * rhs.y);
  }
}

template <typename T> T point2<T>::norm2() const { return dot(*this); }

template <typename T> double point2<T>::length() const { return std::sqrt(static_cast<double>(norm2())); }

template <typename T> point2<float> point2<T>::normalized() const {
    double len = length();
    rassert(len > 0, 325412341231, len);
    return point2f(x, y) / len;
}

template <typename T> std::string point2<T>::to_string() const {
  std::ostringstream ss;
  ss << "(" << x << ", " << y << ")";
  return ss.str();
}

template <typename T> bool point2<T>::operator==(const point2& rhs) const { return x == rhs.x && y == rhs.y; }

template <typename T> bool point2<T>::operator!=(const point2& rhs) const { return !(*this == rhs); }

template <typename T> std::ostream& operator<<(std::ostream& os, const point2<T>& p) {
  os << p.to_string();
  return os;
}

// Scalar * vector (same scalar type)
point2i operator*(int s, const point2i& p) { return p * s; }
point2f operator*(float s, const point2f& p) { return p * s; }

// Mixed scalar for int vectors: int-vector * float -> float-vector
point2f operator*(const point2i& p, float s) { return point2f(static_cast<float>(p.x) * s, static_cast<float>(p.y) * s); }

point2f operator/(const point2i& p, float s) {
  rassert(std::fabs(s) > 0.0f, "division by zero");
  return point2f(static_cast<float>(p.x) / s, static_cast<float>(p.y) / s);
}

point2f operator*(float s, const point2i& p) { return p * s; }

// Explicit instantiations (avoid recompiling template code in every TU)
template struct point2<float>;
template struct point2<int>;
template std::ostream& operator<<(std::ostream& os, const point2<float>& p);
template std::ostream& operator<<(std::ostream& os, const point2<int>& p);
