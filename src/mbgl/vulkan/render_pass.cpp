#include <mbgl/vulkan/render_pass.hpp>

#include <mbgl/vulkan/command_encoder.hpp>
#include <mbgl/vulkan/renderable_resource.hpp>
#include <mbgl/vulkan/context.hpp>
#include <mbgl/util/logging.hpp>

namespace mbgl {
namespace vulkan {

RenderPass::RenderPass(CommandEncoder& commandEncoder_,
                       const char* name,
                       const gfx::RenderPassDescriptor& descriptor_,
                       Context& context)
    : descriptor(descriptor_),
      commandEncoder(commandEncoder_) {
    auto& resource = descriptor.renderable.getResource<RenderableResource>();
    resource.bind();

    std::array<vk::ClearValue, 2> clearValues;

    if (descriptor.clearColor.has_value())
        clearValues[0].setColor(descriptor.clearColor.value().operator std::array<float, 4>());
    clearValues[1].depthStencil.setDepth(descriptor.clearDepth.value_or(1.0f));
    clearValues[1].depthStencil.setStencil(descriptor.clearStencil.value_or(0));

    const auto renderPassBeginInfo = vk::RenderPassBeginInfo()
                                         .setRenderPass(resource.getRenderPass().get())
                                         .setFramebuffer(resource.getFramebuffer().get())
                                         .setRenderArea({{0, 0}, resource.getExtent()})
                                         .setClearValues(clearValues);

    pushDebugGroup(/*render thread*/ {}, name);

    commandEncoder.getPrimaryCommandBuffer()->beginRenderPass(
        renderPassBeginInfo,
        context.getRenderThreadCount() ? vk::SubpassContents::eSecondaryCommandBuffers : vk::SubpassContents::eInline);

    context.performCleanup();
}

RenderPass::~RenderPass() {
    endEncoding();

    popDebugGroup(/*render thread*/ {});
}

void RenderPass::endEncoding() {
    commandEncoder.endEncoding();
}

void RenderPass::clearStencil(std::optional<std::size_t> threadIndex, uint32_t value) const {
    const auto& resource = descriptor.renderable.getResource<RenderableResource>();
    const auto& extent = resource.getExtent();

    const auto attach = vk::ClearAttachment()
                            .setAspectMask(vk::ImageAspectFlagBits::eStencil)
                            .setClearValue(vk::ClearDepthStencilValue(0.0f, value));

    const auto rect = vk::ClearRect().setBaseArrayLayer(0).setLayerCount(1).setRect(
        {{0, 0}, {extent.width, extent.height}});

    commandEncoder.getCommandBuffer(threadIndex)->clearAttachments(attach, rect);
}

void RenderPass::pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) {
    commandEncoder.pushDebugGroup(threadIndex, name);
}

void RenderPass::popDebugGroup(std::optional<std::size_t> threadIndex) {
    commandEncoder.popDebugGroup(threadIndex);
}

void RenderPass::addDebugSignpost(std::optional<std::size_t> threadIndex, const char* name) {
    auto& buffer = commandEncoder.getCommandBuffer(threadIndex);
    commandEncoder.getContext().getBackend().insertDebugLabel(buffer.get(), name);
}

} // namespace vulkan
} // namespace mbgl
