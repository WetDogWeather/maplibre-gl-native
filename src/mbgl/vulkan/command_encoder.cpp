#include <mbgl/vulkan/command_encoder.hpp>
#include <mbgl/vulkan/context.hpp>
#include <mbgl/vulkan/renderable_resource.hpp>
#include <mbgl/vulkan/upload_pass.hpp>
#include <mbgl/vulkan/render_pass.hpp>

#include <cstring>

namespace mbgl {
namespace vulkan {

CommandEncoder::CommandEncoder(Context& context_)
    : context(context_) {}

std::unique_ptr<gfx::UploadPass> CommandEncoder::createUploadPass(const char* name, gfx::Renderable& renderable) {
    return std::make_unique<UploadPass>(renderable, *this, name);
}

std::unique_ptr<gfx::RenderPass> CommandEncoder::createRenderPass(const char* name,
                                                                  const gfx::RenderPassDescriptor& descriptor) {
    auto renderPass = std::make_unique<RenderPass>(*this, name, descriptor, context);
    renderPassInfo = renderPass->getInfo();
    return renderPass;
}

const vk::UniqueCommandBuffer& CommandEncoder::getCommandBuffer(std::int32_t layerIndex) {
    return context.getCommandBuffer(layerIndex, renderPassInfo);
}

void CommandEncoder::present(gfx::Renderable& renderable) {
    renderable.getResource<RenderableResource>().swap();
}

void CommandEncoder::endEncoding() const {
    context.endEncoding();
}

void CommandEncoder::pushDebugGroup(std::int32_t layerIndex, const char* name) {
    pushDebugGroup(layerIndex, name, {});
}

void CommandEncoder::pushDebugGroup(std::int32_t layerIndex, const char* name, const std::array<float, 4>& color) {
    context.getBackend().beginDebugLabel(getCommandBuffer(layerIndex).get(), name, color);
}

void CommandEncoder::popDebugGroup(std::int32_t layerIndex) {
    context.getBackend().endDebugLabel(getCommandBuffer(layerIndex).get());
}

} // namespace vulkan
} // namespace mbgl
