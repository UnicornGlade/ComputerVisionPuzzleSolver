#pragma once

#include <type_traits>
#include <utility>
#include <vector>

// filter: keeps elements x for which pred(x) == true.
template <class T, class Pred>
std::vector<T> filter_vector(const std::vector<T>& input, Pred&& pred) {
    std::vector<T> out;
    out.reserve(input.size());

    for (const auto& x : input) {
        if (pred(x)) out.push_back(x);
    }
    return out;
}

// map: transforms each element x -> f(x) and collects results.
template <class T, class F>
auto map_vector(const std::vector<T>& input, F&& f)
    -> std::vector<std::decay_t<decltype(f(std::declval<const T&>()))>> {
    using U = std::decay_t<decltype(f(std::declval<const T&>()))>;
    std::vector<U> out;
    out.reserve(input.size());

    for (const auto& x : input) {
        out.push_back(f(x));
    }
    return out;
}
