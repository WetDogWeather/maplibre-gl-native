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

    pushDebugGroup(0, name);

    context.performCleanup();
}

RenderPass::~RenderPass() {
    endEncoding();

    popDebugGroup(0);
}

void RenderPass::endEncoding() {
    commandEncoder.endEncoding();
}

void RenderPass::clearStencil(std::int32_t layerIndex, uint32_t value) const {
    const auto& resource = descriptor.renderable.getResource<RenderableResource>();
    const auto& extent = resource.getExtent();

    const auto attach = vk::ClearAttachment()
                            .setAspectMask(vk::ImageAspectFlagBits::eStencil)
                            .setClearValue(vk::ClearDepthStencilValue(0.0f, value));

    const auto rect = vk::ClearRect().setBaseArrayLayer(0).setLayerCount(1).setRect(
        {{0, 0}, {extent.width, extent.height}});

    commandEncoder.getCommandBuffer(layerIndex)->clearAttachments(attach, rect);
}

void RenderPass::pushDebugGroup(std::int32_t layerIndex, const char* name) {
    commandEncoder.pushDebugGroup(layerIndex, name);
}

void RenderPass::popDebugGroup(std::int32_t layerIndex) {
    commandEncoder.popDebugGroup(layerIndex);
}

void RenderPass::addDebugSignpost(std::int32_t layerIndex, const char* name) {
    commandEncoder.getContext().getBackend().insertDebugLabel(commandEncoder.getCommandBuffer(layerIndex).get(), name);
}

void RenderPass::bindVertex(const BufferResource& buf, std::size_t offset, std::size_t, std::size_t size) {
    [[maybe_unused]] const auto actualSize = size ? size : buf.getSizeInBytes() - offset;
    assert(actualSize <= buf.getSizeInBytes());
}

void RenderPass::bindFragment(const BufferResource& buf, std::size_t offset, std::size_t, std::size_t size) {
    [[maybe_unused]] const auto actualSize = size ? size : buf.getSizeInBytes() - offset;
    assert(actualSize <= buf.getSizeInBytes());
}

} // namespace vulkan
} // namespace mbgl
