#pragma once

#include <mbgl/gfx/command_encoder.hpp>

namespace mbgl {
namespace gl {

class Context;

class CommandEncoder final : public gfx::CommandEncoder {
public:
    explicit CommandEncoder(gl::Context& context_)
        : context(context_) {}

    ~CommandEncoder() override;

    friend class UploadPass;
    friend class RenderPass;

    std::unique_ptr<gfx::UploadPass> createUploadPass(const char* name, gfx::Renderable&) override;
    std::unique_ptr<gfx::RenderPass> createRenderPass(const char* name, const gfx::RenderPassDescriptor&) override;
    void present(gfx::Renderable&) override;

private:
    void pushDebugGroup(std::optional<std::size_t> threadIndex, const char* name) override;
    void popDebugGroup(std::optional<std::size_t> threadIndex) override;

public:
    gl::Context& context;
};

} // namespace gl
} // namespace mbgl
