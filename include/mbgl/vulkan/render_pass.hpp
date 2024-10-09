#pragma once

#include <mbgl/gfx/render_pass.hpp>
#include <mbgl/vulkan/renderer_backend.hpp>

#include <memory>
#include <optional>

namespace mbgl {
namespace vulkan {

class BufferResource;
class CommandEncoder;
class Context;

class RenderPass final : public gfx::RenderPass {
public:
    RenderPass(CommandEncoder&, const char* name, const gfx::RenderPassDescriptor&, Context& context);
    ~RenderPass() override;

    CommandEncoder& getEncoder() { return commandEncoder; }
    const gfx::RenderPassDescriptor& getDescriptor() { return descriptor; }
    void endEncoding();

    void clearStencil(std::int32_t layerIndex, uint32_t value = 0) const;

    void addDebugSignpost(std::optional<std::int32_t> layerIndex, const char* name) override;

    void bindVertex(const BufferResource&, std::size_t offset, std::size_t index, std::size_t size = 0);
    void bindFragment(const BufferResource&, std::size_t offset, std::size_t index, std::size_t size = 0);

private:
    void pushDebugGroup(std::optional<std::int32_t> layerIndex, const char* name) override;
    void popDebugGroup(std::optional<std::int32_t> layerIndex) override;

private:
    gfx::RenderPassDescriptor descriptor;
    CommandEncoder& commandEncoder;
};

} // namespace vulkan
} // namespace mbgl
