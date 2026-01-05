#include "fast_random.h"

#include <algorithm>

FastRandom::FastRandom(uint32_t seed) : state_(mixSeed(seed)) {}

uint32_t FastRandom::mixSeed(uint32_t x) {
    // Fast 32-bit mixing (based on Murmur/avalanche style)
    x += 0x9E3779B9u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

uint32_t FastRandom::nextU32() {
    // xorshift32: very fast, decent enough for non-crypto use
    uint32_t x = state_;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state_ = x;
    return x;
}

int32_t FastRandom::nextInt() { return static_cast<int32_t>(nextU32()); }

int32_t FastRandom::nextInt(int32_t minVal, int32_t maxVal) {
    if (minVal > maxVal)
        std::swap(minVal, maxVal);

    // Map uint32 -> [0, range) using multiply-high (fast, low-bias)
    uint32_t range = static_cast<uint32_t>(
        static_cast<uint64_t>(static_cast<uint32_t>(maxVal) - static_cast<uint32_t>(minVal)) + 1ull);

    // If range == 0, it means full uint32 range (min=INT32_MIN, max=INT32_MAX). Just return nextInt().
    if (range == 0)
        return nextInt();

    uint32_t r = nextU32();
    uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(r) * range) >> 32);
    return static_cast<int32_t>(static_cast<uint32_t>(minVal) + v);
}

float FastRandom::nextFloat() {
    // Use 24 high bits -> exact step for float mantissa
    constexpr float inv = 1.0f / 16777216.0f; // 2^24
    return static_cast<float>(nextU32() >> 8) * inv;
}

float FastRandom::nextFloat(float minVal, float maxVal) {
    if (minVal > maxVal)
        std::swap(minVal, maxVal);
    return minVal + (maxVal - minVal) * nextFloat();
}
