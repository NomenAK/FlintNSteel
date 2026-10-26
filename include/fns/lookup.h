// include/fns/lookup.h - prompt lookup draft: the continuation of the last earlier occurrence of the final n-gram
// of the history, n = 4 down to 2.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fns {

// Writes at most `want` ids into out and returns their count (0: no occurrence).
inline int lookup_draft(const std::vector<int32_t> & h, int want, int32_t * out) {
    const int n_h = (int) h.size();
    for (int n = 4; n >= 2; --n) {
        if (n_h < n + 1) continue;
        for (int s = n_h - n - 1; s >= 0; --s) {
            if (!std::equal(h.begin() + s, h.begin() + s + n, h.end() - n)) continue;
            int k = 0;
            while (k < want && s + n + k < n_h) {
                out[k] = h[s + n + k];
                ++k;
            }
            if (k > 0) return k;
        }
    }
    return 0;
}

}  // namespace fns
