#pragma once

#include <mbgl/gfx/command_encoder.hpp>
#include <mbgl/gfx/render_pass.hpp>
#include <mbgl/util/containers.hpp>
#include <mbgl/vulkan/renderer_backend.hpp>

namespace mbgl {
namespace gfx {
class Renderable;
} // namespace gfx

namespace vulkan {

class Context;
class RenderPass;
class UploadPass;

class CommandEncoder final : public gfx::CommandEncoder {
public:
    explicit CommandEncoder(Context& context_);
    ~CommandEncoder() override = default;

    vulkan::Context& getContext() { return context; }
    const vulkan::Context& getContext() const { return context; }

    /// The primary command buffer which contains the render pass and the secondary buffers
    const vk::UniqueCommandBuffer& getPrimaryCommandBuffer() const;
    /// The secondary command buffer used for uploads
    const vk::UniqueCommandBuffer& getUploadCommandBuffer() const;
    /// The secondary command buffer for a given layer, which are encoded, in order, into the primary command buffer
    const vk::UniqueCommandBuffer& getSecondaryCommandBuffer(std::size_t threadIndex);
    const vk::UniqueCommandBuffer& getCommandBuffer(std::optional<std::size_t> threadIndex);

    std::unique_ptr<gfx::UploadPass> createUploadPass(const char* name, gfx::Renderable&) override;
    std::unique_ptr<gfx::RenderPass> createRenderPass(const char* name, const gfx::RenderPassDescriptor&) override;

    void present(gfx::Renderable&) override;

    void endEncoding() const;

private:
    void pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) override;
    void pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name, const std::array<float, 4>& color);
    void popDebugGroup(std::optional<std::size_t> threadIndex) override;

private:
    friend class RenderPass;
    friend class UploadPass;

    vulkan::Context& context;
};

} // namespace vulkan
} // namespace mbgl
