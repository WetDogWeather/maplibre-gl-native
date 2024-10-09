#pragma once

#include <mbgl/gfx/debug_group.hpp>

#include <memory>
#include <string>

namespace mbgl {
namespace gfx {

class RenderPassDescriptor;
class RenderPass;
class Renderable;
class UploadPass;

class CommandEncoder {
protected:
    explicit CommandEncoder() = default;

    friend class DebugGroup<CommandEncoder>;
    virtual void pushDebugGroup(std::optional<std::int32_t> layerIndex, const char* name) = 0;
    virtual void popDebugGroup(std::optional<std::int32_t> layerIndex) = 0;

public:
    virtual ~CommandEncoder() = default;
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;

    DebugGroup<CommandEncoder> createDebugGroup(std::optional<std::int32_t> layerIndex, const char* name) {
        return {*this, layerIndex, name};
    }
    DebugGroup<CommandEncoder> createDebugGroup(std::optional<std::int32_t> layerIndex, std::string_view name) {
        return createDebugGroup(layerIndex, name.data());
    }

    virtual std::unique_ptr<UploadPass> createUploadPass(const char* name, Renderable&) = 0;
    virtual std::unique_ptr<RenderPass> createRenderPass(const char* name, const RenderPassDescriptor&) = 0;
    virtual void present(Renderable&) = 0;

    void endEncoding();
};

} // namespace gfx
} // namespace mbgl
