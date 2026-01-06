#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>

class PriorityQueue {
public:
    struct Top {
        int   value;
        float priority;
    };

    PriorityQueue() = default;

    // Reserve memory for faster bulk usage
    void reserve(std::size_t n);

    bool empty() const noexcept { return heap_.empty(); }
    std::size_t size() const noexcept { return heap_.size(); }

    bool contains(int value) const noexcept;

    // Insert a new value. Returns false if value already exists.
    bool push(int value, float priority);

    // Get current top (min by priority, tie-breaker by smaller value).
    Top top() const;

    // Remove and return current top.
    Top pop();

    // Update priority for existing value. Returns false if value not found.
    bool updatePriority(int value, float newPriority);

    // For tests/debug: checks heap property + index map consistency.
    bool checkInvariants() const noexcept;

private:
    struct Node {
        int   value;
        float priority;
    };

    // Compare nodes for min-heap with deterministic tie-breaker.
    static bool less(const Node& a, const Node& b) noexcept;

    void siftUp(std::size_t i) noexcept;
    void siftDown(std::size_t i) noexcept;

    void swapNodes(std::size_t a, std::size_t b) noexcept;

    std::vector<Node> heap_;
    std::unordered_map<int, std::size_t> pos_;
};
