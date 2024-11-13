#pragma once

#include <cstdint>
#include <optional>

namespace mbgl {
namespace gfx {

template <typename T>
class DebugGroup {
public:
    DebugGroup(T& scope_, std::optional<std::size_t> threadIndex_, const char* name)
        : scope(&scope_),
          threadIndex(threadIndex_) {
        scope->pushDebugGroup(threadIndex, name);
    }

    DebugGroup(std::optional<std::size_t> threadIndex_, DebugGroup&& rhs) noexcept
        : scope(rhs.scope),
          threadIndex(threadIndex_) {
        rhs.scope = nullptr;
    }

    DebugGroup(DebugGroup&& other)
        : scope(other.scope),
          threadIndex(other.threadIndex) {
        other.scope = nullptr;
    }

    DebugGroup(const DebugGroup&) = delete;

    ~DebugGroup() {
        if (scope) {
            scope->popDebugGroup(threadIndex);
        }
    }

private:
    T* scope;
    std::optional<std::size_t> threadIndex;
};

} // namespace gfx
} // namespace mbgl
