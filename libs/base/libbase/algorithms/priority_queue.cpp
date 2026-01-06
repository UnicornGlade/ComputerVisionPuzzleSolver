#include "priority_queue.h"

#include <stdexcept>
#include <utility>

void PriorityQueue::reserve(std::size_t n) {
    heap_.reserve(n);
    pos_.reserve(n);
}

bool PriorityQueue::contains(int value) const noexcept {
    return pos_.find(value) != pos_.end();
}

bool PriorityQueue::less(const Node& a, const Node& b) noexcept {
    // Min by priority, tie-break by smaller value for determinism.
    if (a.priority < b.priority) return true;
    if (b.priority < a.priority) return false;
    return a.value < b.value;
}

bool PriorityQueue::push(int value, float priority) {
    if (contains(value)) return false;

    std::size_t idx = heap_.size();
    heap_.push_back(Node{value, priority});
    pos_[value] = idx;
    siftUp(idx);
    return true;
}

PriorityQueue::Top PriorityQueue::top() const {
    if (heap_.empty()) throw std::runtime_error("PriorityQueue::top() on empty queue");
    const Node& n = heap_.front();
    return Top{n.value, n.priority};
}

PriorityQueue::Top PriorityQueue::pop() {
    if (heap_.empty()) throw std::runtime_error("PriorityQueue::pop() on empty queue");

    Node out = heap_.front();
    pos_.erase(out.value);

    if (heap_.size() == 1) {
        heap_.pop_back();
        return Top{out.value, out.priority};
    }

    // Move last to root
    heap_.front() = heap_.back();
    pos_[heap_.front().value] = 0;
    heap_.pop_back();

    siftDown(0);
    return Top{out.value, out.priority};
}

bool PriorityQueue::updatePriority(int value, float newPriority) {
    auto it = pos_.find(value);
    if (it == pos_.end()) return false;

    std::size_t i = it->second;
    float old = heap_[i].priority;
    heap_[i].priority = newPriority;

    // Decide direction
    if (newPriority < old) {
        siftUp(i);
    } else if (old < newPriority) {
        siftDown(i);
    } else {
        // Same priority; tie-breaker by value unchanged -> no movement needed.
    }
    return true;
}

void PriorityQueue::swapNodes(std::size_t a, std::size_t b) noexcept {
    std::swap(heap_[a], heap_[b]);
    pos_[heap_[a].value] = a;
    pos_[heap_[b].value] = b;
}

void PriorityQueue::siftUp(std::size_t i) noexcept {
    while (i > 0) {
        std::size_t p = (i - 1) / 2;
        if (!less(heap_[i], heap_[p])) break;
        swapNodes(i, p);
        i = p;
    }
}

void PriorityQueue::siftDown(std::size_t i) noexcept {
    const std::size_t n = heap_.size();
    while (true) {
        std::size_t l = 2 * i + 1;
        if (l >= n) break;
        std::size_t r = l + 1;

        std::size_t best = l;
        if (r < n && less(heap_[r], heap_[l])) best = r;

        if (!less(heap_[best], heap_[i])) break;
        swapNodes(i, best);
        i = best;
    }
}

bool PriorityQueue::checkInvariants() const noexcept {
    // Check heap property
    for (std::size_t i = 0; i < heap_.size(); ++i) {
        std::size_t l = 2 * i + 1;
        std::size_t r = l + 1;
        if (l < heap_.size() && less(heap_[l], heap_[i])) return false;
        if (r < heap_.size() && less(heap_[r], heap_[i])) return false;
    }

    // Check pos_ consistency
    if (pos_.size() != heap_.size()) return false;
    for (std::size_t i = 0; i < heap_.size(); ++i) {
        auto it = pos_.find(heap_[i].value);
        if (it == pos_.end()) return false;
        if (it->second != i) return false;
    }
    return true;
}
