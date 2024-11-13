#pragma once

#include <mbgl/vulkan/uniform_buffer.hpp>
#include <mbgl/renderer/layer_group.hpp>

#include <optional>

namespace mbgl {

class PaintParameters;

namespace vulkan {

class RenderPass;

/**
 A layer group for tile-based drawables
 */
class TileLayerGroup : public mbgl::TileLayerGroup {
public:
    TileLayerGroup(int32_t layerIndex, std::size_t initialCapacity, std::string name);

    void upload(gfx::UploadPass&, PaintParameters&) override;
    void render(RenderOrchestrator&, PaintParameters&) override;

    const gfx::UniformBufferArray& getUniformBuffers() const override { return uniformBuffers; };
    gfx::UniformBufferArray& mutableUniformBuffers() override { return uniformBuffers; };

    void preRender(RenderOrchestrator&, PaintParameters&) override;

protected:
    UniformBufferArray uniformBuffers;
};

} // namespace vulkan
} // namespace mbgl
