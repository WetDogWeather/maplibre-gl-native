#pragma once

#include <mbgl/renderer/render_pass.hpp>
#include <mbgl/renderer/render_light.hpp>
#include <mbgl/renderer/render_source.hpp>
#include <mbgl/map/mode.hpp>
#include <mbgl/map/transform_state.hpp>
#include <mbgl/gfx/depth_mode.hpp>
#include <mbgl/gfx/stencil_mode.hpp>
#include <mbgl/gfx/color_mode.hpp>
#include <mbgl/util/mat4.hpp>

#include <array>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace mbgl {

class UpdateParameters;
class RenderStaticData;
class Programs;
class TransformState;
class ImageManager;
class LineAtlas;
class PatternAtlas;
class UnwrappedTileID;

namespace gfx {
class Context;
class RendererBackend;
class CommandEncoder;
class RenderPass;
class ShaderRegistry;
} // namespace gfx

class TransformParameters {
public:
    TransformParameters(const TransformState&);
    mat4 projMatrix;
    mat4 alignedProjMatrix;
    mat4 nearClippedProjMatrix;
    const TransformState state;
};

class PaintParameters {
public:
    PaintParameters(gfx::Context&,
                    float pixelRatio,
                    gfx::RendererBackend&,
                    const EvaluatedLight&,
                    MapMode,
                    MapDebugOptions,
                    TimePoint,
                    const TransformParameters&,
                    RenderStaticData&,
                    LineAtlas&,
                    PatternAtlas&,
                    std::uint64_t frameCount,
                    std::size_t renderThreadCount);
    PaintParameters(PaintParameters&&);

    // N.B.: Copies can be made, but they cannot outlive the original instance
    // This is for use only when encoding in parallel, where each thread needs a separate copy
    explicit PaintParameters(PaintParameters&);

    ~PaintParameters() = default;

    gfx::Context& context;
    gfx::RendererBackend& backend;

    const TransformParameters& transformParams;
    const TransformState& state;
    const EvaluatedLight& evaluatedLight;

    RenderStaticData& staticData;
    LineAtlas& lineAtlas;
    PatternAtlas& patternAtlas;

    RenderPass pass = RenderPass::Opaque;
    MapMode mapMode;
    MapDebugOptions debugOptions;
    TimePoint timePoint;

    float pixelRatio;
    std::array<float, 2> pixelsToGLUnits;

    // Programs is, in effect, an immutable shader registry
    Programs& programs;
    // We're migrating to a dynamic one
    gfx::ShaderRegistry& shaders;

    const std::unique_ptr<gfx::CommandEncoder>& getEncoder() const;
    void setEncoder(std::unique_ptr<gfx::CommandEncoder>&&);

    const std::unique_ptr<gfx::RenderPass>& getRenderPass() const;
    void setRenderPass(std::unique_ptr<gfx::RenderPass>&&);

    gfx::DepthMode depthModeForSublayer(uint8_t n, gfx::DepthMaskType) const;
    gfx::DepthMode depthModeFor3D() const;
    gfx::ColorMode colorModeForRenderPass() const;

    mat4 matrixForTile(const UnwrappedTileID&, bool aligned = false) const;

    // Stencil handling
    void renderTileClippingMasks(std::optional<std::size_t> threadIndex, const RenderTiles&);

    /// Clear the stencil buffer, even if there are no tile masks (for 3D)
    void clearStencil(std::optional<std::size_t> threadIndex);

    /// @brief Get a stencil mode for rendering constrined to the specified tile ID.
    /// The tile ID must have been present in the set previously passed to `renderTileClippingMasks`
    gfx::StencilMode stencilModeForClipping(const UnwrappedTileID&) const;

    /// @brief Initialize a stencil mode for 3D rendering.
    /// @details Clears the tile stencil masks, so `stencilModeForClipping`
    ///          cannot be used until `renderTileClippingMasks` is called again.
    /// @return The stencil mode, each value is unique.
    gfx::StencilMode stencilModeFor3D(std::optional<std::size_t> threadIndex);

private:
    // This needs to be an ordered map so that we have the same order as the renderTiles.
    std::map<UnwrappedTileID, int32_t> tileClippingMaskIDs;
    int32_t nextStencilID = 1;

    std::unique_ptr<gfx::CommandEncoder> encoder;
    std::unique_ptr<gfx::RenderPass> renderPass;

    std::optional<std::reference_wrapper<PaintParameters>> baseParameters;

public:
    uint32_t currentLayer = 0;
    float depthRangeSize = 0.0f;
    uint32_t opaquePassCutoff = 0;
    float symbolFadeChange = 0.0f;
    const uint64_t frameCount;
    std::size_t renderThreadCount;
    std::optional<std::size_t> renderThreadIndex;

    static constexpr int numSublayers = 3;
#if MLN_RENDER_BACKEND_OPENGL
    static constexpr float depthEpsilon = 1.0f / (1 << 16);
#else
    static constexpr float depthEpsilon = 1.0f / (1 << 12);
#endif
    static constexpr int maxStencilValue = 255;
};

} // namespace mbgl
