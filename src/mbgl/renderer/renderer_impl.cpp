#include <mbgl/renderer/renderer_impl.hpp>

#include <mbgl/geometry/line_atlas.hpp>
#include <mbgl/gfx/backend_scope.hpp>
#include <mbgl/gfx/context.hpp>
#include <mbgl/gfx/cull_face_mode.hpp>
#include <mbgl/gfx/render_pass.hpp>
#include <mbgl/gfx/renderer_backend.hpp>
#include <mbgl/gfx/renderable.hpp>
#include <mbgl/gfx/upload_pass.hpp>
#include <mbgl/programs/programs.hpp>
#include <mbgl/renderer/paint_parameters.hpp>
#include <mbgl/renderer/pattern_atlas.hpp>
#include <mbgl/renderer/renderer_observer.hpp>
#include <mbgl/renderer/render_static_data.hpp>
#include <mbgl/renderer/render_tree.hpp>
#include <mbgl/util/convert.hpp>
#include <mbgl/util/string.hpp>
#include <mbgl/util/logging.hpp>
#include <mbgl/util/instrumentation.hpp>

#if MLN_DRAWABLE_RENDERER
#include <mbgl/gfx/drawable_tweaker.hpp>
#include <mbgl/renderer/layer_tweaker.hpp>
#include <mbgl/renderer/render_target.hpp>

#include <limits>
#endif // MLN_DRAWABLE_RENDERER

#if MLN_RENDER_BACKEND_METAL
#include <mbgl/mtl/renderer_backend.hpp>
#include <Metal/MTLCaptureManager.hpp>
#include <Metal/MTLCaptureScope.hpp>
/// Enable programmatic Metal frame captures for specific frame numbers.
/// Requries iOS 13
constexpr auto EnableMetalCapture = 0;
constexpr auto CaptureFrameStart = 0; // frames are 0-based
constexpr auto CaptureFrameCount = 1;
#elif MLN_RENDER_BACKEND_OPENGL
#include <mbgl/gl/defines.hpp>
#if MLN_DRAWABLE_RENDERER
#include <mbgl/gl/drawable_gl.hpp>
#endif // MLN_DRAWABLE_RENDERER
#endif // !MLN_RENDER_BACKEND_METAL

namespace mbgl {

using namespace style;

namespace {

RendererObserver& nullObserver() {
    static RendererObserver observer;
    return observer;
}

} // namespace

Renderer::Impl::Impl(gfx::RendererBackend& backend_,
                     float pixelRatio_,
                     const std::optional<std::string>& localFontFamily_)
    : orchestrator(
          !backend_.contextIsShared(), backend_.getThreadPool(), backend_.getRenderThreadPool(), localFontFamily_),
      backend(backend_),
      observer(&nullObserver()),
      pixelRatio(pixelRatio_) {}

Renderer::Impl::~Impl() {
    assert(gfx::BackendScope::exists());
}

void Renderer::Impl::onPreCompileShader(shaders::BuiltIn shaderID,
                                        gfx::Backend::Type type,
                                        const std::string& additionalDefines) {
    observer->onPreCompileShader(shaderID, type, additionalDefines);
}

void Renderer::Impl::onPostCompileShader(shaders::BuiltIn shaderID,
                                         gfx::Backend::Type type,
                                         const std::string& additionalDefines) {
    observer->onPostCompileShader(shaderID, type, additionalDefines);
}

void Renderer::Impl::onShaderCompileFailed(shaders::BuiltIn shaderID,
                                           gfx::Backend::Type type,
                                           const std::string& additionalDefines) {
    observer->onShaderCompileFailed(shaderID, type, additionalDefines);
}

void Renderer::Impl::setObserver(RendererObserver* observer_) {
    observer = observer_ ? observer_ : &nullObserver();
}

void Renderer::Impl::render(const RenderTree& renderTree,
                            [[maybe_unused]] const std::shared_ptr<UpdateParameters>& updateParameters) {
    MLN_TRACE_FUNC();
    auto& context = backend.getContext();
    context.setObserver(this);

#if MLN_RENDER_BACKEND_METAL
    if constexpr (EnableMetalCapture) {
        const auto& mtlBackend = static_cast<mtl::RendererBackend&>(backend);

        const auto& mtlDevice = mtlBackend.getDevice();

        if (!commandCaptureScope) {
            if (const auto& cmdQueue = mtlBackend.getCommandQueue()) {
                if (const auto captureManager = NS::RetainPtr(MTL::CaptureManager::sharedCaptureManager())) {
                    if ((commandCaptureScope = NS::TransferPtr(captureManager->newCaptureScope(cmdQueue.get())))) {
                        const auto label = "Renderer::Impl frame=" + util::toString(frameCount);
                        commandCaptureScope->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
                        captureManager->setDefaultCaptureScope(commandCaptureScope.get());
                    }
                }
            }
        }

        // "When you capture a frame programmatically, you can capture Metal commands that span multiple
        //  frames by using a custom capture scope. For example, by calling begin() at the start of frame
        //  1 and end() after frame 3, the trace will contain command data from all the buffers that were
        //  committed in the three frames."
        // https://developer.apple.com/documentation/metal/debugging_tools/capturing_gpu_command_data_programmatically
        if constexpr (0 < CaptureFrameStart && 0 < CaptureFrameCount) {
            if (commandCaptureScope) {
                const auto captureManager = NS::RetainPtr(MTL::CaptureManager::sharedCaptureManager());
                if (frameCount == CaptureFrameStart) {
                    constexpr auto captureDest = MTL::CaptureDestination::CaptureDestinationDeveloperTools;
                    if (captureManager && !captureManager->isCapturing() &&
                        captureManager->supportsDestination(captureDest)) {
                        if (auto captureDesc = NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init())) {
                            captureDesc->setCaptureObject(mtlDevice.get());
                            captureDesc->setDestination(captureDest);
                            NS::Error* errorPtr = nullptr;
                            if (captureManager->startCapture(captureDesc.get(), &errorPtr)) {
                                Log::Warning(Event::Render, "Capture Started");
                            } else {
                                std::string errStr = "<none>";
                                if (auto error = NS::TransferPtr(errorPtr)) {
                                    if (auto str = error->localizedDescription()) {
                                        if (auto cstr = str->utf8String()) {
                                            errStr = cstr;
                                        }
                                    }
                                }
                                Log::Warning(Event::Render, "Capture Failed: " + errStr);
                            }
                        }
                    }
                }
            }
        }
        if (commandCaptureScope) {
            commandCaptureScope->beginScope();

            const auto captureManager = NS::RetainPtr(MTL::CaptureManager::sharedCaptureManager());
            if (captureManager->isCapturing()) {
                Log::Info(Event::Render, "Capturing frame " + util::toString(frameCount));
            }
        }
    }
#endif // MLN_RENDER_BACKEND_METAL

    // Blocks execution until the renderable is available.
    backend.getDefaultRenderable().wait();
    context.beginFrame();

    if (!staticData) {
        staticData = std::make_unique<RenderStaticData>(pixelRatio, std::make_unique<gfx::ShaderRegistry>());

        // Initialize legacy shader programs
        staticData->programs.registerWith(*staticData->shaders);

#if MLN_DRAWABLE_RENDERER
        // Initialize shaders for drawables
        const auto programParameters = ProgramParameters{pixelRatio, false};
        backend.initShaders(*staticData->shaders, programParameters);
#endif

        // Notify post-shader registration
        observer->onRegisterShaders(*staticData->shaders);
    }

    const auto& renderTreeParameters = renderTree.getParameters();
    staticData->has3D = renderTreeParameters.has3D;
    staticData->backendSize = backend.getDefaultRenderable().getSize();

    if (renderState == RenderState::Never) {
        observer->onWillStartRenderingMap();
    }

    observer->onWillStartRenderingFrame();

    auto* renderThreadPool = backend.getRenderThreadPool();
    PaintParameters parameters{context,
                               pixelRatio,
                               backend,
                               renderTreeParameters.light,
                               renderTreeParameters.mapMode,
                               renderTreeParameters.debugOptions,
                               renderTreeParameters.timePoint,
                               renderTreeParameters.transformParams,
                               *staticData,
                               renderTree.getLineAtlas(),
                               renderTree.getPatternAtlas(),
                               frameCount,
                               renderThreadPool ? renderThreadPool->getThreadCount() : 0};

    parameters.symbolFadeChange = renderTreeParameters.symbolFadeChange;
    parameters.opaquePassCutoff = renderTreeParameters.opaquePassCutOff;

    // Make a copy of the paint parameters for each render thread to use
    const auto layerCount = orchestrator.numLayerGroups();
    std::vector<PaintParameters> threadParameters;
    threadParameters.reserve(parameters.renderThreadCount);
    for (std::size_t i = 0; i < parameters.renderThreadCount; ++i) {
        threadParameters.emplace_back(parameters);
        threadParameters.back().renderThreadIndex = i;
    }

    const auto defaultDepthRangeSize = 1.0f - static_cast<float>(layerCount + 2) * PaintParameters::numSublayers *
                                                  PaintParameters::depthEpsilon;

    const auto& sourceRenderItems = renderTree.getSourceRenderItems();

#if MLN_DRAWABLE_RENDERER
    const auto& layerRenderItems = renderTree.getLayerRenderItemMap();
#else
    const auto& layerRenderItems = renderTree.getLayerRenderItems();
#endif

#if MLN_DRAWABLE_RENDERER
    mbgl::unordered_map<std::thread::id, gfx::BackendScope> threadScopes;
    std::mutex threadScopesMutex;
    const auto initScope = [&] {
        if (!gfx::BackendScope::exists()) {
            std::unique_lock<std::mutex> lock(threadScopesMutex);
            threadScopes.try_emplace(
                std::this_thread::get_id(), parameters.backend, gfx::BackendScope::ScopeType::Implicit);
        }
    };
#endif

    // - UPLOAD PASS -------------------------------------------------------------------------------
    // Uploads all required buffers and images before we do any actual rendering.
    {
        const auto uploadPass = parameters.getEncoder()->createUploadPass("upload",
                                                                          parameters.backend.getDefaultRenderable());
#if !defined(NDEBUG)
        const auto debugGroup = uploadPass->createDebugGroup(/*render thread*/ {}, "upload");
#endif

        // Update all clipping IDs + upload buckets.
        for (const RenderItem& item : sourceRenderItems) {
            item.upload(*uploadPass);
        }
        for (const RenderItem& item : layerRenderItems) {
            item.upload(*uploadPass);
        }
        staticData->upload(*uploadPass);
        renderTree.getLineAtlas().upload(*uploadPass);
        renderTree.getPatternAtlas().upload(*uploadPass);
    }

#if MLN_DRAWABLE_RENDERER
    // - LAYER GROUP UPDATE ------------------------------------------------------------------------
    // Updates all layer groups and process changes
    if (staticData && staticData->shaders) {
        orchestrator.updateLayers(
            *staticData->shaders, context, renderTreeParameters.transformParams.state, updateParameters, renderTree);
    }

    orchestrator.processChanges();

    const auto getParams = [&](const std::optional<std::size_t> threadIndex) -> PaintParameters& {
        return threadIndex ? threadParameters[*threadIndex] : parameters;
    };
    // Upload layer groups
    {
        MLN_TRACE_ZONE(upload);
        const auto uploadPass = parameters.getEncoder()->createUploadPass("layerGroup-upload",
                                                                          parameters.backend.getDefaultRenderable());
#if !defined(NDEBUG)
        const auto debugGroup = uploadPass->createDebugGroup(/*render thread*/ {}, "layerGroup-upload");
#endif

        // Tweakers are run in the upload pass so they can set up uniforms.
        orchestrator.visitLayerGroups(
            renderThreadPool,
            [&](auto& layerGroup, const std::optional<std::size_t> threadIndex, const std::size_t layerIndex) {
                MLN_TRACE_ZONE(tweak);
                auto& params = getParams(threadIndex);
                params.currentLayer = layerIndex;
                layerGroup.runTweakers(renderTree, params);
        });
        orchestrator.visitDebugLayerGroups(
            [&](LayerGroupBase& layerGroup) { layerGroup.runTweakers(renderTree, parameters); });

        // Update the debug layer groups
        orchestrator.updateDebugLayerGroups(renderTree, parameters);

        // Give the layers a chance to upload
        orchestrator.visitLayerGroups(renderThreadPool, [&](auto& layerGroup, const auto threadIndex, auto layerIndex) {
            MLN_TRACE_ZONE(upload);
            initScope();
            auto& params = getParams(threadIndex);
            params.currentLayer = layerIndex;
            layerGroup.upload(*uploadPass, params);
        });

        // Give the render targets a chance to upload
        orchestrator.visitRenderTargets(
            [&](RenderTarget& renderTarget) { renderTarget.upload(*uploadPass, parameters); });

        // Upload the Debug layer group
        orchestrator.visitDebugLayerGroups(
            [&](LayerGroupBase& layerGroup) { layerGroup.upload(*uploadPass, parameters); });
    }

    const Size atlasSize = parameters.patternAtlas.getPixelSize();
    const auto& worldSize = parameters.staticData.backendSize;
    const shaders::GlobalPaintParamsUBO globalPaintParamsUBO = {
        /* .pattern_atlas_texsize = */ {static_cast<float>(atlasSize.width), static_cast<float>(atlasSize.height)},
        /* .units_to_pixels = */ {1.0f / parameters.pixelsToGLUnits[0], 1.0f / parameters.pixelsToGLUnits[1]},
        /* .world_size = */ {static_cast<float>(worldSize.width), static_cast<float>(worldSize.height)},
        /* .camera_to_center_distance = */ parameters.state.getCameraToCenterDistance(),
        /* .symbol_fade_change = */ parameters.symbolFadeChange,
        /* .aspect_ratio = */ parameters.state.getSize().aspectRatio(),
        /* .pixel_ratio = */ parameters.pixelRatio,
        /* .zoom = */ static_cast<float>(parameters.state.getZoom()),
        /* .pad1 = */ 0,
    };
    auto& globalUniforms = context.mutableGlobalUniformBuffers();
    globalUniforms.createOrUpdate(
        shaders::idGlobalPaintParamsUBO, &globalPaintParamsUBO, context, parameters.renderThreadIndex);
#endif

    // - 3D PASS
    // -------------------------------------------------------------------------------------
    // Renders any 3D layers bottom-to-top to unique FBOs with texture
    // attachments, but share the same depth rbo between them.
    const auto common3DPass = [&] {
        if (parameters.staticData.has3D) {
            parameters.staticData.backendSize = parameters.backend.getDefaultRenderable().getSize();

            const auto debugGroup(parameters.getEncoder()->createDebugGroup(/*render thread*/ {}, "common-3d"));
            parameters.pass = RenderPass::Pass3D;

            // TODO is this needed?
            // if (!parameters.staticData.depthRenderbuffer ||
            //    parameters.staticData.depthRenderbuffer->getSize() != parameters.staticData.backendSize) {
            //    parameters.staticData.depthRenderbuffer =
            //        parameters.context.createRenderbuffer<gfx::RenderbufferPixelType::Depth>(
            //            parameters.staticData.backendSize);
            //}
            // parameters.staticData.depthRenderbuffer->setShouldClear(true);
        }
    };

#if MLN_DRAWABLE_RENDERER
    const auto drawable3DPass = [&] {
        const auto debugGroup(parameters.getEncoder()->createDebugGroup(/*render thread*/ {}, "drawables-3d"));
        assert(parameters.pass == RenderPass::Pass3D);

        // draw layer groups, 3D pass
        orchestrator.visitLayerGroups([&](LayerGroupBase& layerGroup, std::size_t layerIndex) {
            parameters.currentLayer = layerCount - layerIndex - 1;
            layerGroup.render(orchestrator, parameters);
        });
    };
#endif // MLN_DRAWABLE_RENDERER

#if MLN_LEGACY_RENDERER
    const auto renderLayer3DPass = [&] {
        const auto debugGroup(parameters.getEncoder()->createDebugGroup(/*render thread*/ {}, "3d"));
        int32_t i = static_cast<int32_t>(layerRenderItems.size()) - 1;
        for (auto it = layerRenderItems.begin(); it != layerRenderItems.end() && i >= 0; ++it, --i) {
            parameters.currentLayer = i;
            const RenderItem& renderItem = it->get();
            if (renderItem.hasRenderPass(parameters.pass)) {
                const auto layerDebugGroup(parameters.getEncoder()->createDebugGroup({}, renderItem.getName()));
                renderItem.render(parameters);
            }
        }
    };
#endif // MLN_LEGACY_RENDERER

#if MLN_DRAWABLE_RENDERER
    const auto drawableTargetsPass = [&] {
        MLN_TRACE_ZONE(targets);
        // draw render targets
        orchestrator.visitRenderTargets(
            [&](RenderTarget& renderTarget) { renderTarget.render(orchestrator, renderTree, parameters); });
    };
#endif

    const auto commonClearPass = [&] {
        MLN_TRACE_ZONE(clear);
        // - CLEAR
        // -------------------------------------------------------------------------------------
        // Renders the backdrop of the OpenGL view. This also paints in areas where
        // we don't have any tiles whatsoever.
        {
            std::optional<Color> color;
            if (parameters.debugOptions & MapDebugOptions::Overdraw) {
                color = Color::black();
            } else if (!backend.contextIsShared()) {
                color = renderTreeParameters.backgroundColor;
            }
            parameters.setRenderPass(parameters.getEncoder()->createRenderPass(
                "main buffer", {parameters.backend.getDefaultRenderable(), color, 1.0f, 0}));
        }
    };

    // Actually render the layers
#if MLN_DRAWABLE_RENDERER
    // Drawables
    const auto drawableOpaquePass = [&] {
        MLN_TRACE_ZONE(opaque);
        const auto debugGroup(parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "drawables-opaque"));

        // draw layer groups, opaque pass
        orchestrator.visitLayerGroups(
            renderThreadPool, /*reversed=*/true, [&](auto& layerGroup, auto threadIndex, auto layerIndex) {
                MLN_TRACE_ZONE(opaque);
                initScope();
                auto& params = getParams(threadIndex);
                params.currentLayer = layerCount - layerIndex - 1;
                params.pass = RenderPass::Opaque;
                params.depthRangeSize = defaultDepthRangeSize;
                layerGroup.render(orchestrator, params);
            });
    };

    const auto drawableTranslucentPass = [&] {
        MLN_TRACE_ZONE(translucent);
        const auto debugGroup(
            parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "drawables-translucent"));

        // draw layer groups, translucent pass
        orchestrator.visitLayerGroups(renderThreadPool, [&](auto& layerGroup, auto threadIndex, auto layerIndex) {
            MLN_TRACE_ZONE(translucent);
            initScope();
            auto& params = getParams(threadIndex);
            params.currentLayer = layerCount - layerIndex - 1;
            params.pass = RenderPass::Translucent;
            params.depthRangeSize = defaultDepthRangeSize;
            layerGroup.render(orchestrator, params);
        });

        // Finally, render any legacy layers which have not been converted to drawables.
        // Note that they may be out of order, this is just a temporary fix for `RenderLocationIndicatorLayer` (#2216)
        parameters.depthRangeSize = defaultDepthRangeSize;
        int32_t i = static_cast<int32_t>(layerRenderItems.size()) - 1;
        for (auto it = layerRenderItems.begin(); it != layerRenderItems.end() && i >= 0; ++it, --i) {
            parameters.currentLayer = i;
            const RenderItem& item = *it;
            if (item.hasRenderPass(parameters.pass)) {
                item.render(parameters);
            }
        }
    };
#endif

#if MLN_LEGACY_RENDERER
    // Render everything top-to-bottom by using reverse iterators. Render opaque objects first.
    const auto renderLayerOpaquePass = [&] {
        const auto debugGroup(parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "opaque"));
        parameters.pass = RenderPass::Opaque;
        parameters.depthRangeSize = 1 - (layerRenderItems.size() + 2) * PaintParameters::numSublayers *
                                            PaintParameters::depthEpsilon;

        uint32_t i = 0;
        for (auto it = layerRenderItems.rbegin(); it != layerRenderItems.rend(); ++it, ++i) {
            parameters.currentLayer = i;
            const RenderItem& renderItem = it->get();
            if (renderItem.hasRenderPass(parameters.pass)) {
                const auto layerDebugGroup(parameters.getRenderPass()->createDebugGroup({}, renderItem.getName()));
                renderItem.render(parameters);
            }
        }
    };

    // Make a second pass, rendering translucent objects. This time, we render bottom-to-top.
    const auto renderLayerTranslucentPass = [&] {
        const auto debugGroup(parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "translucent"));
        parameters.pass = RenderPass::Translucent;
        parameters.depthRangeSize = 1 - (layerRenderItems.size() + 2) * PaintParameters::numSublayers *
                                            PaintParameters::depthEpsilon;

        int32_t i = static_cast<int32_t>(layerRenderItems.size()) - 1;
        for (auto it = layerRenderItems.begin(); it != layerRenderItems.end() && i >= 0; ++it, --i) {
            parameters.currentLayer = i;
            const RenderItem& renderItem = it->get();
            if (renderItem.hasRenderPass(parameters.pass)) {
                const auto layerDebugGroup(parameters.getRenderPass()->createDebugGroup({}, renderItem.getName()));
                renderItem.render(parameters);
            }
        }
    };
#endif // MLN_LEGACY_RENDERER

#if MLN_DRAWABLE_RENDERER
    const auto drawableDebugOverlays = [&] {
        MLN_TRACE_ZONE(debug);
        // Renders debug overlays.
        {
            const auto debugGroup(parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "debug"));
            orchestrator.visitDebugLayerGroups(
                [&](LayerGroupBase& layerGroup) { layerGroup.render(orchestrator, parameters); });
        }
    };
#endif // MLN_DRAWABLE_RENDERER

#if MLN_LEGACY_RENDERER
    const auto renderDebugOverlays = [&] {
        // Renders debug overlays.
        {
            const auto debugGroup(parameters.getRenderPass()->createDebugGroup(/*render thread*/ {}, "debug"));

            // Finalize the rendering, e.g. by calling debug render calls per tile.
            // This guarantees that we have at least one function per tile called.
            // When only rendering layers via the stylesheet, it's possible that we
            // don't ever visit a tile during rendering.
            for (const RenderItem& renderItem : sourceRenderItems) {
                renderItem.render(parameters);
            }
        }

#if !defined(NDEBUG)
        if (parameters.debugOptions & MapDebugOptions::StencilClip) {
            // Render tile clip boundaries, using stencil buffer to calculate fill color.
            parameters.context.visualizeStencilBuffer();
        } else if (parameters.debugOptions & MapDebugOptions::DepthBuffer) {
            // Render the depth buffer.
            parameters.context.visualizeDepthBuffer(parameters.depthRangeSize);
        }
#endif
    };
#endif // MLN_LEGACY_RENDERER

#if (MLN_DRAWABLE_RENDERER && !MLN_LEGACY_RENDERER)
    orchestrator.visitLayerGroups(
        [&](auto& layerGroup, auto /*layerIndex*/) { layerGroup.preRender(orchestrator, parameters); });

    if (parameters.staticData.has3D) {
        common3DPass();
        drawable3DPass();
    }
    drawableTargetsPass();
    commonClearPass();

    // Bind the globals on each thread/encoder where they will be used
    if (renderThreadPool) {
        renderThreadPool->eachThread([&](auto threadIndex) {
            MLN_TRACE_ZONE(globals);
            context.bindGlobalUniformBuffers(*parameters.getRenderPass(), threadIndex);
        });
    } else {
        context.bindGlobalUniformBuffers(*parameters.getRenderPass(), {});
    }

    drawableOpaquePass();
    drawableTranslucentPass();
    drawableDebugOverlays();
#elif (MLN_LEGACY_RENDERER && !MLN_DRAWABLE_RENDERER)
    if (parameters.staticData.has3D) {
        common3DPass();
        renderLayer3DPass();
    }
    commonClearPass();
    renderLayerOpaquePass();
    renderLayerTranslucentPass();
    renderDebugOverlays();
#else
    static_assert(0, "Must define one of (MLN_DRAWABLE_RENDERER, MLN_LEGACY_RENDERER)");
#endif // MLN_LEGACY_RENDERER

#if MLN_DRAWABLE_RENDERER
    // Give the layers a chance to do cleanup
    orchestrator.visitLayerGroups([&](auto& layerGroup, auto layerIndex) {
        parameters.currentLayer = layerIndex;
        layerGroup.postRender(orchestrator, parameters);
    });
    context.unbindGlobalUniformBuffers(*parameters.getRenderPass(), parameters.renderThreadIndex);
#endif

    // Ends the RenderPass
    parameters.setRenderPass({});

    const auto startRendering = util::MonotonicTimer::now().count();
    {
        MLN_TRACE_ZONE(present);
        // present submits render commands
        parameters.getEncoder()->present(parameters.backend.getDefaultRenderable());
    }
    const auto renderingTime = util::MonotonicTimer::now().count() - startRendering;

    parameters.setEncoder({});
    context.endFrame();

#if MLN_RENDER_BACKEND_METAL
    if constexpr (EnableMetalCapture) {
        if (commandCaptureScope) {
            commandCaptureScope->endScope();

            const auto captureManager = NS::RetainPtr(MTL::CaptureManager::sharedCaptureManager());
            if (frameCount == CaptureFrameStart + CaptureFrameCount - 1 && captureManager->isCapturing()) {
                captureManager->stopCapture();
            }
        }
    }
#endif // MLN_RENDER_BACKEND_METAL

    const auto encodingTime = renderTree.getElapsedTime() - renderingTime;

    observer->onDidFinishRenderingFrame(
        renderTreeParameters.loaded ? RendererObserver::RenderMode::Full : RendererObserver::RenderMode::Partial,
        renderTreeParameters.needsRepaint,
        renderTreeParameters.placementChanged,
        encodingTime,
        renderingTime);

    if (!renderTreeParameters.loaded) {
        renderState = RenderState::Partial;
    } else if (renderState != RenderState::Fully) {
        renderState = RenderState::Fully;
        observer->onDidFinishRenderingMap();
    }

    frameCount += 1;
    MLN_END_FRAME();
}

void Renderer::Impl::reduceMemoryUse() {
    assert(gfx::BackendScope::exists());
    backend.getContext().reduceMemoryUsage();
}

} // namespace mbgl
