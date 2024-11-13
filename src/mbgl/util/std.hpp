#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace mbgl {
namespace util {

template <typename Container, typename ForwardIterator, typename Predicate>
void erase_if(Container& container, ForwardIterator it, Predicate pred) {
    while (it != container.end()) {
        if (pred(*it)) {
            it = container.erase(it);
        } else {
            ++it;
        }
    }
}

template <typename Container, typename Predicate>
void erase_if(Container& container, Predicate pred) {
    erase_if(container, container.begin(), pred);
}

} // namespace util

#if __cplusplus < 202302L
// A literal that's safe to assign/compare with `size_t` on all platforms
// Built-in with C++23
inline constexpr std::size_t operator"" _uz(unsigned long long x) {
    return static_cast<std::size_t>(x);
}
#endif

} // namespace mbgl
