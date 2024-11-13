#include <mbgl/gl/render_pass.hpp>
#include <mbgl/gl/command_encoder.hpp>
#include <mbgl/gl/renderable_resource.hpp>
#include <mbgl/gl/context.hpp>

namespace mbgl {
namespace gl {

RenderPass::RenderPass(gl::CommandEncoder& commandEncoder_,
                       const char* name,
                       const gfx::RenderPassDescriptor& descriptor)
    : commandEncoder(commandEncoder_),
      debugGroup(commandEncoder.createDebugGroup(/*main thread*/ {}, name)) {
    descriptor.renderable.getResource<gl::RenderableResource>().bind();
    const auto clearDebugGroup(commandEncoder.createDebugGroup({}, "clear"));
    commandEncoder.context.clear(descriptor.clearColor, descriptor.clearDepth, descriptor.clearStencil);
}

void RenderPass::pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) {
    commandEncoder.pushDebugGroup(threadIndex, name);
}

void RenderPass::popDebugGroup(std::optional<std::size_t> threadIndex) {
    commandEncoder.popDebugGroup(threadIndex);
}

} // namespace gl
} // namespace mbgl
