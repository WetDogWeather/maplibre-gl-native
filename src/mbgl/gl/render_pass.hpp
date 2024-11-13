#pragma once

#include <mbgl/gfx/render_pass.hpp>

namespace mbgl {
namespace gfx {

class CommandEncoder;

} // namespace gfx

namespace gl {

class CommandEncoder;
class Context;

class RenderPass final : public gfx::RenderPass {
public:
    RenderPass(gl::CommandEncoder&, const char* name, const gfx::RenderPassDescriptor&);

private:
    void pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) override;
    void popDebugGroup(std::optional<std::size_t> threadIndex) override;

private:
    gl::CommandEncoder& commandEncoder;
    const gfx::DebugGroup<gfx::CommandEncoder> debugGroup;
};

} // namespace gl
} // namespace mbgl
