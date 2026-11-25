// argmax_no_eog against a naive reference: EOG tokens never win (also when one is the plain maximum), ties go to the
// first token (across 64-token blocks, inside one, and -0.0 before +0.0), and a maximum in the tail past the last whole
// block wins. Built with the driver's own compile options (CMakeLists.txt), so the vectorised loop is the one tested.
#include "greedy.h"

#include <cstdio>
#include <random>

static int32_t reference(const std::vector<float> & l, const std::vector<char> & eog) {
    int32_t best = -1;
    for (int32_t t = 0; t < int32_t(l.size()); ++t) {
        if (!eog[size_t(t)] && (best < 0 || l[size_t(t)] > l[size_t(best)])) {
            best = t;
        }
    }
    return best;
}

int main() {
    const int32_t     n_vocab = 4099;  // 64 whole blocks and a tail of 3
    std::vector<char> eog(size_t(n_vocab), 0);
    eog[7] = eog[2000] = eog[4095] = 1;
    std::mt19937                       rng(11);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    int                                bad = 0;
    std::vector<float>                 l(size_t(n_vocab), 0.0f);
    for (int round = 0; round < 200; ++round) {
        for (int c = 0; c < 8; ++c) {
            for (auto & x : l) {
                x = u(rng);
            }
            switch (c) {
                case 0: l[7] = 9.0f; break;                          // an EOG maximum
                case 1: l[100] = l[50] = 9.0f; break;                // a tie across blocks: 50 wins
                case 2: l[2000] = 9.0f; l[3] = 8.5f; break;          // EOG first, then 3
                case 3: l[30] = l[20] = 9.0f; break;                 // a tie inside one block: 20 wins
                case 4: l[4097] = 9.0f; break;                       // the maximum in the tail
                case 5:                                              // -0.0 ties +0.0 and comes first: 5 wins
                    for (auto & x : l) {
                        x = std::min(x, -0.5f);
                    }
                    l[5]  = -0.0f;
                    l[70] = 0.0f;
                    break;
                default: break;
            }
            const int32_t got = argmax_no_eog(l.data(), n_vocab, eog), want = reference(l, eog);
            if (got != want) {
                std::fprintf(stderr, "round %d case %d: got %d, want %d\n", round, c, got, want);
                ++bad;
            }
        }
    }
    std::printf("argmax_no_eog: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
