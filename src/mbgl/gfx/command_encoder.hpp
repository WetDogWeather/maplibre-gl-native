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
    virtual void pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) = 0;
    virtual void popDebugGroup(std::optional<std::size_t> threadIndex) = 0;

public:
    virtual ~CommandEncoder() = default;
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;

    DebugGroup<CommandEncoder> createDebugGroup(std::optional<std::size_t> threadIndex, const char* name) {
        return {*this, threadIndex, name};
    }
    DebugGroup<CommandEncoder> createDebugGroup(std::optional<std::size_t> threadIndex, std::string_view name) {
        return {*this, threadIndex, name.data()};
    }

    virtual std::unique_ptr<UploadPass> createUploadPass(const char* name, Renderable&) = 0;
    virtual std::unique_ptr<RenderPass> createRenderPass(const char* name, const RenderPassDescriptor&) = 0;
    virtual void present(Renderable&) = 0;

    void endEncoding();
};

} // namespace gfx
} // namespace mbgl
