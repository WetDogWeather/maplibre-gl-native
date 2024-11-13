#include <mbgl/renderer/paint_parameters.hpp>

#include <mbgl/gfx/command_encoder.hpp>
#include <mbgl/gfx/cull_face_mode.hpp>
#include <mbgl/gfx/render_pass.hpp>
#include <mbgl/map/transform_state.hpp>
#include <mbgl/renderer/render_static_data.hpp>
#include <mbgl/renderer/render_source.hpp>
#include <mbgl/renderer/render_tile.hpp>
#include <mbgl/renderer/update_parameters.hpp>
#include <mbgl/util/convert.hpp>
#include <mbgl/util/logging.hpp>

#if MLN_RENDER_BACKEND_METAL
#include <mbgl/mtl/context.hpp>
#include <mbgl/shaders/mtl/clipping_mask.hpp>
#endif // MLN_RENDER_BACKEND_METAL

#if MLN_RENDER_BACKEND_VULKAN
#include <mbgl/vulkan/render_pass.hpp>
#include <mbgl/shaders/vulkan/clipping_mask.hpp>
#include <mbgl/vulkan/context.hpp>
#endif // MLN_RENDER_BACKEND_VULKAN

namespace mbgl {

TransformParameters::TransformParameters(const TransformState& state_)
    : state(state_) {
    // Update the default matrices to the current viewport dimensions.
    state.getProjMatrix(projMatrix);

    // Also compute a projection matrix that aligns with the current pixel grid,
    // taking into account odd viewport sizes.
    state.getProjMatrix(alignedProjMatrix, 1, true);

    // Calculate a second projection matrix with the near plane moved further,
    // to a tenth of the far value, so as not to waste depth buffer precision on
    // very close empty space, for layer types (fill-extrusion) that use the
    // depth buffer to emulate real-world space.
    state.getProjMatrix(nearClippedProjMatrix, static_cast<uint16_t>(0.1 * state.getCameraToCenterDistance()));
}

PaintParameters::PaintParameters(gfx::Context& context_,
                                 float pixelRatio_,
                                 gfx::RendererBackend& backend_,
                                 const EvaluatedLight& evaluatedLight_,
                                 MapMode mode_,
                                 MapDebugOptions debugOptions_,
                                 TimePoint timePoint_,
                                 const TransformParameters& transformParams_,
                                 RenderStaticData& staticData_,
                                 LineAtlas& lineAtlas_,
                                 PatternAtlas& patternAtlas_,
                                 uint64_t frameCount_,
                                 std::size_t renderThreadCount_)
    : context(context_),
      backend(backend_),
      transformParams(transformParams_),
      state(transformParams_.state),
      evaluatedLight(evaluatedLight_),
      staticData(staticData_),
      lineAtlas(lineAtlas_),
      patternAtlas(patternAtlas_),
      mapMode(mode_),
      debugOptions(debugOptions_),
      timePoint(timePoint_),
      pixelRatio(pixelRatio_),
#ifndef NDEBUG
      programs((debugOptions & MapDebugOptions::Overdraw) ? staticData_.overdrawPrograms : staticData_.programs),
#else
      programs(staticData_.programs),
#endif
      shaders(*staticData_.shaders),
      encoder(context.createCommandEncoder()),
      frameCount(frameCount_),
      renderThreadCount(renderThreadCount_) {
    pixelsToGLUnits = {{2.0f / state.getSize().width, -2.0f / state.getSize().height}};

    if (state.getViewportMode() == ViewportMode::FlippedY) {
        pixelsToGLUnits[1] *= -1;
    }
}

PaintParameters::PaintParameters(PaintParameters&& other)
    : context(other.context),
      backend(other.backend),
      transformParams(other.transformParams),
      state(other.state),
      evaluatedLight(other.evaluatedLight),
      staticData(other.staticData),
      lineAtlas(other.lineAtlas),
      patternAtlas(other.patternAtlas),
      pass(other.pass),
      mapMode(other.mapMode),
      debugOptions(other.debugOptions),
      timePoint(other.timePoint),
      pixelRatio(other.pixelRatio),
      pixelsToGLUnits(other.pixelsToGLUnits),
      programs(other.programs),
      shaders(other.shaders),
      tileClippingMaskIDs(std::move(other.tileClippingMaskIDs)),
      nextStencilID(other.nextStencilID),
      encoder(std::move(other.encoder)),
      renderPass(std::move(other.renderPass)),
      baseParameters(std::move(other.baseParameters)),
      currentLayer(other.currentLayer),
      depthRangeSize(other.depthRangeSize),
      opaquePassCutoff(other.opaquePassCutoff),
      symbolFadeChange(other.symbolFadeChange),
      frameCount(other.frameCount),
      renderThreadCount(other.renderThreadCount),
      renderThreadIndex(other.renderThreadIndex) {}

PaintParameters::PaintParameters(PaintParameters& other)
    : context(other.context),
      backend(other.backend),
      transformParams(other.transformParams),
      state(other.state),
      evaluatedLight(other.evaluatedLight),
      staticData(other.staticData),
      lineAtlas(other.lineAtlas),
      patternAtlas(other.patternAtlas),
      pass(other.pass),
      mapMode(other.mapMode),
      debugOptions(other.debugOptions),
      timePoint(other.timePoint),
      pixelRatio(other.pixelRatio),
      pixelsToGLUnits(other.pixelsToGLUnits),
      programs(other.programs),
      shaders(other.shaders),
      tileClippingMaskIDs(), // not copied
      nextStencilID(1),      // not copied
      encoder(),             // not copied
      renderPass(),          // not copied
      baseParameters(other.baseParameters ? other.baseParameters->get() : other),
      currentLayer(other.currentLayer),
      depthRangeSize(other.depthRangeSize),
      opaquePassCutoff(other.opaquePassCutoff),
      symbolFadeChange(other.symbolFadeChange),
      frameCount(other.frameCount),
      renderThreadCount(other.renderThreadCount),
      renderThreadIndex(other.renderThreadIndex) {}

const std::unique_ptr<gfx::CommandEncoder>& PaintParameters::getEncoder() const {
    return baseParameters ? baseParameters->get().encoder : encoder;
}

void PaintParameters::setEncoder(std::unique_ptr<gfx::CommandEncoder>&& enc) {
    (baseParameters ? baseParameters->get() : *this).encoder = std::move(enc);
}

const std::unique_ptr<gfx::RenderPass>& PaintParameters::getRenderPass() const {
    return baseParameters ? baseParameters->get().renderPass : renderPass;
}

void PaintParameters::setRenderPass(std::unique_ptr<gfx::RenderPass>&& rp) {
    (baseParameters ? baseParameters->get() : *this).renderPass = std::move(rp);
}

mat4 PaintParameters::matrixForTile(const UnwrappedTileID& tileID, bool aligned) const {
    mat4 matrix;
    state.matrixFor(matrix, tileID);
    matrix::multiply(matrix, aligned ? transformParams.alignedProjMatrix : transformParams.projMatrix, matrix);
    return matrix;
}

gfx::DepthMode PaintParameters::depthModeForSublayer([[maybe_unused]] uint8_t n, gfx::DepthMaskType mask) const {
    if (currentLayer < opaquePassCutoff) {
        return gfx::DepthMode::disabled();
    }

#if MLN_RENDER_BACKEND_OPENGL
    float depth = depthRangeSize + ((1 + currentLayer) * numSublayers + n) * depthEpsilon;
    return gfx::DepthMode{gfx::DepthFunctionType::LessEqual, mask, {depth, depth}};
#else
    return gfx::DepthMode{gfx::DepthFunctionType::LessEqual, mask};
#endif
}

gfx::DepthMode PaintParameters::depthModeFor3D() const {
#if MLN_RENDER_BACKEND_OPENGL
    return gfx::DepthMode{gfx::DepthFunctionType::LessEqual, gfx::DepthMaskType::ReadWrite, {0.0, depthRangeSize}};
#else
    return gfx::DepthMode{gfx::DepthFunctionType::LessEqual, gfx::DepthMaskType::ReadWrite};
#endif
}

namespace {

template <typename TIter>
using GetTileIDFunc = const UnwrappedTileID& (*)(const typename TIter::value_type&);

using TileMaskIDMap = std::map<UnwrappedTileID, int32_t>;

// Check whether we can reuse a clip mask for a new set of tiles
bool tileIDsCovered(const RenderTiles& tiles, const TileMaskIDMap& idMap) {
    return idMap.size() == tiles->size() &&
           std::equal(idMap.cbegin(), idMap.cend(), tiles->cbegin(), tiles->cend(), [=](const auto& a, const auto& b) {
               return a.first == b.get().id;
           });
}

} // namespace

void PaintParameters::clearStencil([[maybe_unused]] std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();

    nextStencilID = 1;
    tileClippingMaskIDs.clear();

#if MLN_RENDER_BACKEND_METAL
    auto& mtlContext = static_cast<mtl::Context&>(context);

    // Metal doesn't have an equivalent of `glClear`, so we clear the buffer by drawing zero to (0:0,0)
#if !defined(NDEBUG)
    const auto debugGroup = getRenderPass()->createDebugGroup(threadIndex, "tile-clip-mask-clear");
#endif

    const std::vector<shaders::ClipUBO> tileUBO = {
        shaders::ClipUBO{/*.matrix=*/util::cast<float>(matrixForTile({0, 0, 0})),
                         /*.stencil_ref=*/0,
                         /*.pad=*/0,
                         0,
                         0}};
    mtlContext.renderTileClippingMasks(*getRenderPass(), staticData, tileUBO);
    context.renderingStats().stencilClears++;
#elif MLN_RENDER_BACKEND_VULKAN
    const auto& vulkanRenderPass = static_cast<vulkan::RenderPass&>(*getRenderPass());
    vulkanRenderPass.clearStencil(threadIndex, 0);

    context.renderingStats().stencilClears++;
#else // !MLN_RENDER_BACKEND_METAL
    context.clearStencilBuffer(0b00000000);
#endif
}

void PaintParameters::renderTileClippingMasks(std::optional<std::size_t> threadIndex, const RenderTiles& renderTiles) {
    MLN_TRACE_FUNC();

    // We can avoid updating the mask if it already contains the same set of tiles.
    if (!renderTiles || !getRenderPass() || tileIDsCovered(renderTiles, tileClippingMaskIDs)) {
        return;
    }

    tileClippingMaskIDs.clear();

    // If the stencil value will overflow, clear the target to ensure ensure that none of the new
    // values remain set somewhere in it. Otherwise we can continue to overwrite it incrementally.
    const auto count = renderTiles->size();
    if (nextStencilID + count > maxStencilValue) {
        clearStencil(threadIndex);
    }

#if MLN_RENDER_BACKEND_METAL
    // Assign a stencil ID and build a UBO for each tile in the set
    std::vector<shaders::ClipUBO> tileUBOs;
    for (const auto& tileRef : *renderTiles) {
        const auto& tileID = tileRef.get().id;

        const int32_t stencilID = nextStencilID;
        const auto result = tileClippingMaskIDs.insert(std::make_pair(tileID, stencilID));
        if (result.second) {
            // inserted
            nextStencilID++;
        } else {
            // already present
            continue;
        }

        if (tileUBOs.empty()) {
            tileUBOs.reserve(count);
        }

        tileUBOs.emplace_back(shaders::ClipUBO{/*.matrix=*/util::cast<float>(matrixForTile(tileID)),
                                               /*.stencil_ref=*/static_cast<uint32_t>(stencilID),
                                               /*.pad=*/0,
                                               0,
                                               0});
    }

    if (!tileUBOs.empty()) {
#if !defined(NDEBUG)
        const auto debugGroup = getRenderPass()->createDebugGroup(threadIndex, "tile-clip-masks");
#endif

        auto& mtlContext = static_cast<mtl::Context&>(context);
        mtlContext.renderTileClippingMasks(*getRenderPass(), staticData, tileUBOs);

        mtlContext.renderingStats().stencilUpdates++;
    }

#elif MLN_RENDER_BACKEND_VULKAN

    std::vector<shaders::ClipUBO> tileUBOs;
    for (const auto& tileRef : *renderTiles) {
        const auto& tileID = tileRef.get().id;

        const uint32_t stencilID = nextStencilID;
        const auto result = tileClippingMaskIDs.insert(std::make_pair(tileID, stencilID));
        if (result.second) {
            // inserted
            nextStencilID++;
        } else {
            // already present
            continue;
        }

        if (tileUBOs.empty()) {
            tileUBOs.reserve(count);
        }

        tileUBOs.emplace_back(shaders::ClipUBO{util::cast<float>(matrixForTile(tileID)), stencilID});
    }

    if (!tileUBOs.empty()) {
#if !defined(NDEBUG)
        const auto debugGroup = getRenderPass()->createDebugGroup(threadIndex, "tile-clip-masks");
#endif

        auto& vulkanContext = static_cast<vulkan::Context&>(context);
        vulkanContext.renderTileClippingMasks(threadIndex, *getRenderPass(), staticData, tileUBOs);
        vulkanContext.renderingStats().stencilUpdates++;
    }

#else  // !MLN_RENDER_BACKEND_METAL
    auto program = staticData.shaders->getLegacyGroup().get<ClippingMaskProgram>();

    if (!program) {
        return;
    }

    static_cast<gl::Context&>(context).renderingStats().stencilUpdates++;

    const style::Properties<>::PossiblyEvaluated properties{};
    const ClippingMaskProgram::Binders paintAttributeData(properties, 0);

    for (const auto& tileRef : *renderTiles) {
        const auto& tileID = tileRef.get().id;

        const int32_t stencilID = nextStencilID;
        const auto result = tileClippingMaskIDs.insert(std::make_pair(tileID, stencilID));
        if (result.second) {
            // inserted
            nextStencilID++;
        } else {
            // already present
            continue;
        }

        program->draw(context,
                      *getRenderPass(),
                      gfx::Triangles(),
                      gfx::DepthMode::disabled(),
                      gfx::StencilMode{gfx::StencilMode::Always{},
                                       stencilID,
                                       0b11111111,
                                       gfx::StencilOpType::Keep,
                                       gfx::StencilOpType::Keep,
                                       gfx::StencilOpType::Replace},
                      gfx::ColorMode::disabled(),
                      gfx::CullFaceMode::disabled(),
                      *staticData.quadTriangleIndexBuffer,
                      staticData.clippingMaskSegments,
                      ClippingMaskProgram::computeAllUniformValues(
                          ClippingMaskProgram::LayoutUniformValues{
                              uniforms::matrix::Value(matrixForTile(tileID)),
                          },
                          paintAttributeData,
                          properties,
                          static_cast<float>(state.getZoom())),
                      ClippingMaskProgram::computeAllAttributeBindings(
                          *staticData.tileVertexBuffer, paintAttributeData, properties),
                      ClippingMaskProgram::TextureBindings{},
                      "clipping/" + util::toString(stencilID));
    }
#endif // MLN_RENDER_BACKEND_METAL
}

gfx::StencilMode PaintParameters::stencilModeForClipping(const UnwrappedTileID& tileID) const {
    auto it = tileClippingMaskIDs.find(tileID);
    assert(it != tileClippingMaskIDs.end());
    const int32_t id = it != tileClippingMaskIDs.end() ? it->second : 0b00000000;
    return gfx::StencilMode{gfx::StencilMode::Equal{0b11111111},
                            id,
                            0b00000000,
                            gfx::StencilOpType::Keep,
                            gfx::StencilOpType::Keep,
                            gfx::StencilOpType::Replace};
}

gfx::StencilMode PaintParameters::stencilModeFor3D(std::optional<std::size_t> threadIndex) {
    if (nextStencilID + 1 > maxStencilValue) {
        clearStencil(threadIndex);
    }

    // We're potentially destroying the stencil clipping mask in this pass. That
    // means we'll have to recreate it for the next source if any.
    tileClippingMaskIDs.clear();

    const int32_t id = nextStencilID++;
    return gfx::StencilMode{gfx::StencilMode::NotEqual{0b11111111},
                            id,
                            0b11111111,
                            gfx::StencilOpType::Keep,
                            gfx::StencilOpType::Keep,
                            gfx::StencilOpType::Replace};
}

gfx::ColorMode PaintParameters::colorModeForRenderPass() const {
    if (debugOptions & MapDebugOptions::Overdraw) {
        constexpr float overdraw = 1.0f / 8.0f;
        return gfx::ColorMode{
            gfx::ColorMode::Add{gfx::ColorBlendFactorType::ConstantColor, gfx::ColorBlendFactorType::One},
            Color{overdraw, overdraw, overdraw, 0.0f},
            gfx::ColorMode::Mask{true, true, true, true}};
    } else if (pass == RenderPass::Translucent) {
        return gfx::ColorMode::alphaBlended();
    } else {
        return gfx::ColorMode::unblended();
    }
}

} // namespace mbgl
