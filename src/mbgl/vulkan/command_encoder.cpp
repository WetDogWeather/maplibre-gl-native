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
    return std::make_unique<RenderPass>(*this, name, descriptor, context);
}

const vk::UniqueCommandBuffer& CommandEncoder::getPrimaryCommandBuffer() const {
    return context.getPrimaryCommandBuffer();
}

const vk::UniqueCommandBuffer& CommandEncoder::getUploadCommandBuffer() const {
    return context.getUploadCommandBuffer();
}

const vk::UniqueCommandBuffer& CommandEncoder::getSecondaryCommandBuffer(std::size_t threadIndex) {
    return context.getSecondaryCommandBuffer(threadIndex);
}

const vk::UniqueCommandBuffer& CommandEncoder::getCommandBuffer(std::optional<std::size_t> threadIndex) {
    return context.getCommandBuffer(threadIndex);
}

void CommandEncoder::present(gfx::Renderable& renderable) {
    renderable.getResource<RenderableResource>().swap();
}

void CommandEncoder::endEncoding() const {
    context.endEncoding();
}

void CommandEncoder::pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) {
    pushDebugGroup(threadIndex, name, {});
}

void CommandEncoder::pushDebugGroup(std::optional<std::size_t> threadIndex,
                                    const char* name,
                                    const std::array<float, 4>& color) {
    MLN_TRACE_FUNC();
    context.getBackend().beginDebugLabel(getCommandBuffer(threadIndex).get(), name, color);
}

void CommandEncoder::popDebugGroup(std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();
    context.getBackend().endDebugLabel(getCommandBuffer(threadIndex).get());
}

} // namespace vulkan
} // namespace mbgl
