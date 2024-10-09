#pragma once

#include <cstdint>
#include <optional>

namespace mbgl {
namespace gfx {

template <typename T>
class DebugGroup {
public:
    DebugGroup(T& scope_, std::optional<std::int32_t> layerIndex_, const char* name)
        : scope(&scope_),
          layerIndex(layerIndex_) {
        scope->pushDebugGroup(layerIndex, name);
    }

    DebugGroup(std::optional<std::int32_t> layerIndex_, DebugGroup&& rhs) noexcept
        : scope(rhs.scope),
          layerIndex(layerIndex_) {
        rhs.scope = nullptr;
    }

    DebugGroup(DebugGroup&& other)
        : scope(other.scope),
          layerIndex(other.layerIndex) {
        other.scope = nullptr;
    }

    DebugGroup(const DebugGroup&) = delete;

    ~DebugGroup() {
        if (scope) {
            scope->popDebugGroup(layerIndex);
        }
    }

private:
    T* scope;
    std::optional<std::int32_t> layerIndex;
};

} // namespace gfx
} // namespace mbgl
