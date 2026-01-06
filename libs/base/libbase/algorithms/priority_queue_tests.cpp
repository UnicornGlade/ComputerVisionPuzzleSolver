// PriorityQueue_gtest.cpp
#include "priority_queue.h"

#include <gtest/gtest.h>
#include <set>
#include <random>
#include <vector>

TEST(PriorityQueue, BasicOrder) {
    PriorityQueue q;
    EXPECT_TRUE(q.empty());

    EXPECT_TRUE(q.push(10, 5.0f));
    EXPECT_TRUE(q.push(20, 2.0f));
    EXPECT_TRUE(q.push(30, 3.0f));
    EXPECT_EQ(q.size(), 3u);
    EXPECT_TRUE(q.checkInvariants());

    auto t = q.top();
    EXPECT_EQ(t.value, 20);
    EXPECT_FLOAT_EQ(t.priority, 2.0f);

    auto p1 = q.pop();
    EXPECT_EQ(p1.value, 20);

    auto p2 = q.pop();
    EXPECT_EQ(p2.value, 30);

    auto p3 = q.pop();
    EXPECT_EQ(p3.value, 10);

    EXPECT_TRUE(q.empty());
}

TEST(PriorityQueue, UpdatePriority) {
    PriorityQueue q;
    ASSERT_TRUE(q.push(1, 10.0f));
    ASSERT_TRUE(q.push(2, 20.0f));
    ASSERT_TRUE(q.push(3, 30.0f));

    // Decrease key
    EXPECT_TRUE(q.updatePriority(3, 1.0f));
    EXPECT_TRUE(q.checkInvariants());
    EXPECT_EQ(q.top().value, 3);

    // Increase key
    EXPECT_TRUE(q.updatePriority(3, 100.0f));
    EXPECT_TRUE(q.checkInvariants());
    EXPECT_EQ(q.top().value, 1);

    // Missing key
    EXPECT_FALSE(q.updatePriority(999, 0.0f));
}

TEST(PriorityQueue, TieBreakerByValue) {
    PriorityQueue q;
    ASSERT_TRUE(q.push(5, 1.0f));
    ASSERT_TRUE(q.push(3, 1.0f));
    ASSERT_TRUE(q.push(4, 1.0f));
    ASSERT_TRUE(q.checkInvariants());

    EXPECT_EQ(q.pop().value, 3);
    EXPECT_EQ(q.pop().value, 4);
    EXPECT_EQ(q.pop().value, 5);
}

TEST(PriorityQueue, RejectDuplicates) {
    PriorityQueue q;
    EXPECT_TRUE(q.push(7, 1.0f));
    EXPECT_FALSE(q.push(7, 2.0f));
    EXPECT_EQ(q.size(), 1u);

    auto t = q.top();
    EXPECT_EQ(t.value, 7);
    EXPECT_FLOAT_EQ(t.priority, 1.0f);
}

TEST(PriorityQueue, Stress_BigWithUpdates) {
    constexpr int N = 100'000;
    constexpr int STEPS = 10'000;          // pop-steps
    constexpr int UPDATES_PER_STEP = 100;  // updates per step

    PriorityQueue q;
    q.reserve(static_cast<std::size_t>(N));

    // Reference ordered by (priority, value)
    std::set<std::pair<float, int>> ref;
    std::vector<float> prio(N);

    std::mt19937 rng(123456u);
    std::uniform_real_distribution<float> distP(0.0f, 1.0f);
    std::uniform_int_distribution<int> distId(0, N - 1);

    for (int i = 0; i < N; ++i) {
        float p = distP(rng);
        prio[i] = p;
        ASSERT_TRUE(q.push(i, p));
        ref.insert({p, i});
    }
    ASSERT_TRUE(q.checkInvariants());
    ASSERT_EQ(q.size(), static_cast<std::size_t>(N));
    ASSERT_FALSE(ref.empty());

    auto assert_top_equal = [&]() {
        auto t = q.top();
        auto it = ref.begin();
        ASSERT_NE(it, ref.end());
        ASSERT_EQ(t.value, it->second);
        ASSERT_FLOAT_EQ(t.priority, it->first);
    };

    assert_top_equal();

    for (int step = 0; step < STEPS; ++step) {
        int applied = 0;
        int attempts = 0;

        while (applied < UPDATES_PER_STEP && attempts < UPDATES_PER_STEP * 50) {
            ++attempts;
            int id = distId(rng);
            if (!q.contains(id)) continue;

            float newP = distP(rng);

            auto itOld = ref.find({prio[id], id});
            ASSERT_NE(itOld, ref.end());
            ref.erase(itOld);

            prio[id] = newP;
            ref.insert({newP, id});

            ASSERT_TRUE(q.updatePriority(id, newP));
            ++applied;
        }
        ASSERT_EQ(applied, UPDATES_PER_STEP);

        if ((step & 1023) == 0) {
            ASSERT_TRUE(q.checkInvariants());
        }

        assert_top_equal();

        auto got = q.pop();
        auto it = ref.begin();
        ASSERT_NE(it, ref.end());
        ASSERT_EQ(got.value, it->second);
        ASSERT_FLOAT_EQ(got.priority, it->first);
        ref.erase(it);
    }

    ASSERT_EQ(q.size(), ref.size());
    ASSERT_TRUE(q.checkInvariants());
}
