#pragma once

#include <cstdint>

class FastRandom {
public:
    explicit FastRandom(uint32_t seed=239);

    // Returns a random 32-bit unsigned integer
    uint32_t nextU32();

    // Returns a random signed int (full range)
    int32_t nextInt();

    // Returns a random int in [minVal, maxVal] (inclusive)
    int32_t nextInt(int32_t minVal, int32_t maxVal);

    // Returns a random float in [0, 1)
    float nextFloat();

    // Returns a random float in [minVal, maxVal)
    float nextFloat(float minVal, float maxVal);

private:
    static uint32_t mixSeed(uint32_t x);

    uint32_t state_;
};
