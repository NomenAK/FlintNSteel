// Host adapter for FlintNSteel: no torch dependency.
#pragma once
#include <sstream>
#include <stdexcept>
template<class... T> inline void exl3_throw(const T&... t) {
    std::ostringstream s;
    (s << ... << t);
    throw std::runtime_error(s.str());
}
#define TORCH_CHECK(c, ...) do { if (!(c)) exl3_throw(__VA_ARGS__); } while (0)
