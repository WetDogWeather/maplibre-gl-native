#pragma once

#include <cstdint>

namespace mbgl {
namespace gfx {

template <typename T>
class DebugGroup {
public:
    DebugGroup(T& scope_, std::int32_t layerIndex_, const char* name)
        : scope(&scope_),
          layerIndex(layerIndex_) {
        scope->pushDebugGroup(layerIndex, name);
    }

    DebugGroup(std::int32_t layerIndex_, DebugGroup&& rhs) noexcept
        : scope(rhs.scope),
          layerIndex(layerIndex_) {
        rhs.scope = nullptr;
    }

    ~DebugGroup() {
        if (scope) {
            scope->popDebugGroup(layerIndex);
        }
    }

private:
    T* scope;
    std::int32_t layerIndex;
};

} // namespace gfx
} // namespace mbgl
