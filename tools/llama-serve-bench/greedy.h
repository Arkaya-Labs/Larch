// Greedy sampling with end-of-generation tokens excluded (llama-server's ignore_eos biases them to -inf), shared by
// llama-serve-bench and its check against a naive reference (greedy_test.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// x as an int32 in the same order as the floats, -0.0 equal to +0.0 so ties stay ties: without fast-math an int max
// vectorises, a float max does not (GCC 15 not at all, Clang 21 in part). The fold is integer-only (the negatives move
// up one, so -0.0 lands on +0.0's 0), which no float flag such as -ffast-math can optimise away.
inline int32_t order_key(float x) {
    int32_t b;
    std::memcpy(&b, &x, sizeof b);
    return (b ^ ((b >> 31) & 0x7fffffff)) - (b >> 31);
}

// A plain argmax, first best on ties: a llama_sampler copies the whole vocabulary per token. Each 64-float block's
// max key is one vectorised loop, then the first block holding the maximum is searched for it: built for wsl5070's
// CPU, 12 us per 152k-token row just copied from the GPU, the speed of reading it (20 us on baseline x86-64, 80 for
// std::max_element). The masked scan runs only when an EOG token is the plain maximum. NaN logits (a broken decode)
// make the pick unspecified.
inline int32_t argmax_no_eog(const float * l, int32_t n_vocab, const std::vector<char> & eog) {
    constexpr int32_t B   = 64;
    int32_t           key = INT32_MIN, from = 0, i = 0;
    for (; i + B <= n_vocab; i += B) {
        int32_t m = INT32_MIN;
        for (int32_t j = 0; j < B; ++j) {
            m = std::max(m, order_key(l[i + j]));
        }
        if (m > key) {
            key  = m;
            from = i;
        }
    }
    for (; i < n_vocab; ++i) {
        if (order_key(l[i]) > key) {
            key  = order_key(l[i]);
            from = i;
        }
    }
    auto best = from;
    while (order_key(l[best]) != key) {
        ++best;
    }
    if (!eog[size_t(best)]) {
        return best;
    }
    best = -1;
    for (int32_t t = 0; t < n_vocab; ++t) {
        if (!eog[size_t(t)] && (best < 0 || l[t] > l[best])) {
            best = t;
        }
    }
    return best;
}
