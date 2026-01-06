#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace stats {

template <typename T>
concept AllowedType = std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, std::size_t> || std::is_same_v<T, std::uint8_t>;

// Throws std::invalid_argument if values is empty.
template <AllowedType T> T minValue(const std::vector<T> &values);

template <AllowedType T> T maxValue(const std::vector<T> &values);

// Median / percentile return double (because interpolation).
// Throws std::invalid_argument if values is empty.
template <AllowedType T> double median(const std::vector<T> &values);

// p in [0, 100]. Linear interpolation on sorted data.
// Throws std::invalid_argument if values is empty or p out of range.
template <AllowedType T> double percentile(const std::vector<T> &values, double p);

// "N values - [v0, v1, v2, v3, v4, ... vN-5, vN-4, vN-3, vN-2, vN-1]"
// If N <= 10: list all values.
// If N == 0: "0 values - []"
template <AllowedType T> std::string previewValues(const std::vector<T> &values);

// Summary:
// - for int/size_t/uint8_t: "N values - (min=... 10%=... median=... 90%=... max=...)"
// - for float/double: same, but with fixed decimals (default 2)
template <AllowedType T>
    requires(!std::is_floating_point_v<T>)
std::string summaryStats(const std::vector<T> &values);

std::string summaryStats(const std::vector<float> &values, int decimals = 2);
std::string summaryStats(const std::vector<double> &values, int decimals = 2);

} // namespace stats
