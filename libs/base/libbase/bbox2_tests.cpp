#include <libbase/bbox2.h>

#include <gtest/gtest.h>

TEST(bbox2, Empty) {
    bbox2i b;
    EXPECT_TRUE(b.is_empty());
    EXPECT_EQ(b.width(), 0);
    EXPECT_EQ(b.height(), 0);
}

TEST(bbox2, IncludePixelHalfOpen) {
    bbox2i b;
    b.include_pixel(10, 20);
    EXPECT_FALSE(b.is_empty());
    EXPECT_EQ(b.min.x, 10);
    EXPECT_EQ(b.min.y, 20);
    EXPECT_EQ(b.max.x, 11);
    EXPECT_EQ(b.max.y, 21);
    EXPECT_EQ(b.width(), 1);
    EXPECT_EQ(b.height(), 1);

    b.include_pixel(12, 22);
    EXPECT_EQ(b.min.x, 10);
    EXPECT_EQ(b.min.y, 20);
    EXPECT_EQ(b.max.x, 13);
    EXPECT_EQ(b.max.y, 23);
    EXPECT_EQ(b.width(), 3);
    EXPECT_EQ(b.height(), 3);

    EXPECT_TRUE(b.contains_pixel(10, 20));
    EXPECT_TRUE(b.contains_pixel(12, 22));
    EXPECT_FALSE(b.contains_pixel(13, 22));
    EXPECT_FALSE(b.contains_pixel(12, 23));
}

TEST(bbox2, IncludeBox) {
    bbox2i a;
    a.include_pixel(0, 0);
    a.include_pixel(1, 1); // => [0,2)x[0,2)

    bbox2i b;
    b.include_pixel(10, 10); // => [10,11)x[10,11)

    a.include_box(b);
    EXPECT_EQ(a.min.x, 0);
    EXPECT_EQ(a.min.y, 0);
    EXPECT_EQ(a.max.x, 11);
    EXPECT_EQ(a.max.y, 11);
}
