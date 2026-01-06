#include <gtest/gtest.h>

#include <sstream>
#include <type_traits>

#include <libbase/runtime_assert.h>

#include "point2.h"

using libbase::point2f;
using libbase::point2i;

TEST(point2, AddSubInt) {
  const point2i a(1, 2);
  const point2i b(10, -5);

  const point2i c = a + b;
  EXPECT_EQ(c.x, 11);
  EXPECT_EQ(c.y, -3);

  const point2i d = a - b;
  EXPECT_EQ(d.x, -9);
  EXPECT_EQ(d.y, 7);
}

TEST(point2, ScalarMulDivInt) {
  const point2i a(3, 4);

  const point2i b = a * 2;
  EXPECT_EQ(b.x, 6);
  EXPECT_EQ(b.y, 8);

  const point2i c = a / 2; // integer division
  EXPECT_EQ(c.x, 1);
  EXPECT_EQ(c.y, 2);
}

TEST(point2, IntDivByFloatGivesFloatPoint) {
  const point2i a(3, 4);

  auto b = a / 2.0f;
  static_assert(std::is_same_v<std::remove_cv_t<decltype(b)>, point2f>);

  EXPECT_FLOAT_EQ(b.x, 1.5f);
  EXPECT_FLOAT_EQ(b.y, 2.0f);
}

TEST(point2, IntMulByFloatGivesFloatPoint) {
  const point2i a(3, 4);

  auto b = a * 0.5f;
  static_assert(std::is_same_v<std::remove_cv_t<decltype(b)>, point2f>);

  EXPECT_FLOAT_EQ(b.x, 1.5f);
  EXPECT_FLOAT_EQ(b.y, 2.0f);

  auto c = 0.5f * a;
  EXPECT_FLOAT_EQ(c.x, 1.5f);
  EXPECT_FLOAT_EQ(c.y, 2.0f);
}

TEST(point2, DotAndNorm2Float) {
  const point2f a(1.0f, 2.0f);
  const point2f b(3.0f, 4.0f);

  EXPECT_FLOAT_EQ(a.dot(b), 11.0f);
  EXPECT_FLOAT_EQ(a.norm2(), 5.0f);
}

TEST(point2, DotIntOverflowThrows) {
  // 46341^2 > INT32_MAX, so dot with itself must fail.
  const point2i a(46341, 0);
  EXPECT_THROW((void)a.norm2(), ::assertion_error);

  const point2i b(46341, 46341);
  EXPECT_THROW((void)b.dot(b), ::assertion_error);
}

TEST(point2, BracketAccess) {
  point2i a(7, 9);
  a[0] = 1;
  a[1] = 2;
  EXPECT_EQ(a.x, 1);
  EXPECT_EQ(a.y, 2);

  EXPECT_THROW((void)a[2], ::assertion_error);
}

TEST(point2, ToStringAndOstream) {
  const point2i a(1, 2);
  EXPECT_EQ(a.to_string(), "(1, 2)");

  std::ostringstream ss;
  ss << a;
  EXPECT_EQ(ss.str(), "(1, 2)");
}

TEST(point2, Equality) {
  EXPECT_TRUE(point2i(1, 2) == point2i(1, 2));
  EXPECT_TRUE(point2i(1, 2) != point2i(2, 1));
}
