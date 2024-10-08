#pragma once

#include <mbgl/gfx/command_encoder.hpp>
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
    const vk::UniqueCommandBuffer& getCommandBuffer(std::int32_t layerIndex);

    std::unique_ptr<gfx::UploadPass> createUploadPass(const char* name, gfx::Renderable&) override;
    std::unique_ptr<gfx::RenderPass> createRenderPass(const char* name, const gfx::RenderPassDescriptor&) override;

    void setRenderPassInfo(vk::RenderPassBeginInfo info) { renderPassInfo.emplace(std::move(info)); }

    void present(gfx::Renderable&) override;

    void endEncoding() const;

private:
    void pushDebugGroup(std::int32_t layerIndex, const char* name) override;
    void pushDebugGroup(std::int32_t layerIndex, const char* name, const std::array<float, 4>& color);
    void popDebugGroup(std::int32_t layerIndex) override;

private:
    friend class RenderPass;
    friend class UploadPass;

    vulkan::Context& context;
    std::optional<vk::RenderPassBeginInfo> renderPassInfo;
};

} // namespace vulkan
} // namespace mbgl
