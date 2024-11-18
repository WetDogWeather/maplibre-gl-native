#include <mbgl/vulkan/context.hpp>

#include <mbgl/gfx/shader_registry.hpp>
#include <mbgl/programs/program_parameters.hpp>
#include <mbgl/renderer/paint_parameters.hpp>
#include <mbgl/renderer/render_static_data.hpp>
#include <mbgl/renderer/render_target.hpp>
#include <mbgl/vulkan/command_encoder.hpp>
#include <mbgl/vulkan/drawable_builder.hpp>
#include <mbgl/vulkan/offscreen_texture.hpp>
#include <mbgl/vulkan/layer_group.hpp>
#include <mbgl/vulkan/tile_layer_group.hpp>
#include <mbgl/vulkan/renderable_resource.hpp>
#include <mbgl/vulkan/render_pass.hpp>
#include <mbgl/vulkan/texture2d.hpp>
#include <mbgl/vulkan/vertex_attribute.hpp>
#include <mbgl/shaders/vulkan/shader_program.hpp>
#include <mbgl/shaders/vulkan/clipping_mask.hpp>
#include <mbgl/util/hash.hpp>
#include <mbgl/util/logging.hpp>
#include <mbgl/util/thread_pool.hpp>
#include <mbgl/util/traits.hpp>

#include <glslang/Public/ShaderLang.h>

#include <algorithm>
#include <cstring>

namespace mbgl {
namespace vulkan {

// Maximum number of vertex attributes, per vertex descriptor
// 32 on most devices (~30% Android use 16),
// per https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxVertexInputBindings
// this can be queried at runtime (VkPhysicalDeviceLimits.maxVertexInputBindings)
constexpr uint32_t maximumVertexBindingCount = 16;

constexpr uint32_t globalDescriptorPoolSize = 3 * 4;
constexpr uint32_t layerDescriptorPoolSize = 3 * 256;
constexpr uint32_t drawableUniformDescriptorPoolSize = 3 * 1024;
constexpr uint32_t drawableImageDescriptorPoolSize = drawableUniformDescriptorPoolSize / 2;

static uint32_t glslangRefCount = 0;

class RenderbufferResource : public gfx::RenderbufferResource {
public:
    RenderbufferResource() = default;
};

Context::Context(RendererBackend& backend_)
    : gfx::Context(vulkan::maximumVertexBindingCount),
      backend(backend_),
      renderThreadCount(backend.getRenderThreadCount()),
      globalUniformBuffers(DescriptorSetType::Global, 0, shaders::globalUBOCount),
      descriptorPoolMaps(renderThreadCount + 1) {
    if (glslangRefCount++ == 0) {
        glslang::InitializeProcess();
    }

    initFrameResources();
}

Context::~Context() noexcept {
    backend.getThreadPool().runRenderJobs(true /* closeQueue */);

    destroyResources();

    if (--glslangRefCount == 0) {
        glslang::FinalizeProcess();
    }
}

void Context::initFrameResources() {
    using namespace shaders;

    const auto& device = backend.getDevice();
    const auto frameCount = backend.getMaxFrames();

    // Reduce the pool sizes somewhat when using multiple threads
    const std::uint32_t poolSizeMultiplier = renderThreadCount ? 2 : 1;
    const std::uint32_t poolSizeDivisor = renderThreadCount ? renderThreadCount : 1;

    // One set of descriptor pools for the primary render thread, and one for each secondary render thread
    for (auto i = 0_uz; i <= renderThreadCount; ++i) {
        auto& map = descriptorPoolMaps[i];

        map.reserve(underlying_type(DescriptorSetType::Count));
        map.emplace(DescriptorSetType::Global,
                    DescriptorPoolGrowable{globalDescriptorPoolSize * poolSizeMultiplier / poolSizeDivisor, static_cast<uint32_t>(globalUBOCount)});
        map.emplace(DescriptorSetType::Layer,
                    DescriptorPoolGrowable{layerDescriptorPoolSize * poolSizeMultiplier / poolSizeDivisor, static_cast<uint32_t>(maxUBOCountPerLayer)});
        map.emplace(
            DescriptorSetType::DrawableUniform,
            DescriptorPoolGrowable{drawableUniformDescriptorPoolSize * poolSizeMultiplier / poolSizeDivisor, static_cast<uint32_t>(maxUBOCountPerDrawable)});
        map.emplace(
            DescriptorSetType::DrawableImage,
            DescriptorPoolGrowable{drawableImageDescriptorPoolSize * poolSizeMultiplier / poolSizeDivisor, static_cast<uint32_t>(maxTextureCountPerShader)});
    }

    // command buffers
    const vk::CommandBufferAllocateInfo primaryAllocateInfo(
        backend.getCommandPool().get(), vk::CommandBufferLevel::ePrimary, frameCount);
    auto primaryCommandBuffers = device->allocateCommandBuffersUnique(primaryAllocateInfo);
    auto uploadCommandBuffers = device->allocateCommandBuffersUnique(primaryAllocateInfo);

    frameResources.reserve(frameCount);

    for (auto frameIndex = 0_uz; frameIndex < frameCount; ++frameIndex) {
        frameResources.emplace_back(renderThreadCount,
                                    primaryCommandBuffers[frameIndex],
                                    uploadCommandBuffers[frameIndex],
                                    device->createSemaphoreUnique({}),
                                    device->createSemaphoreUnique({}),
                                    device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled)));

        auto& frame = frameResources.back();

        eachRenderThread([&](const auto threadIndex) {
            auto& pool = backend.getCommandPool(threadIndex);
            const vk::CommandBufferAllocateInfo secondaryAllocateInfo(
                pool.get(), vk::CommandBufferLevel::eSecondary, 1);
            auto secondaryCommandBuffers = device->allocateCommandBuffersUnique(secondaryAllocateInfo);
            vk::UniqueCommandBuffer& buffer = secondaryCommandBuffers[0];
            backend.setDebugName(
                buffer.get(),
                "SecondaryCommandBuffer_" + util::toString(frameIndex) + "_" + util::toString(threadIndex));
            frame.secondaryCommandBuffers[threadIndex] = std::move(buffer);
        });

        backend.setDebugName(frame.primaryCommandBuffer.get(), "PrimaryCommandBuffer_" + std::to_string(frameIndex));
        backend.setDebugName(frame.uploadCommandBuffer.get(), "UploadCommandBuffer_" + std::to_string(frameIndex));
        backend.setDebugName(frame.frameSemaphore.get(), "FrameSemaphore_" + std::to_string(frameIndex));
        backend.setDebugName(frame.surfaceSemaphore.get(), "SurfaceSemaphore_" + std::to_string(frameIndex));
        backend.setDebugName(frame.flightFrameFence.get(), "FrameFence_" + std::to_string(frameIndex));
    }

    // force placeholder texture upload before any descriptor sets
    (void)getDummyTexture({});

    buildUniformDescriptorSetLayout(
        globalUniformDescriptorSetLayout, globalUBOCount, "GlobalUniformDescriptorSetLayout");
    buildUniformDescriptorSetLayout(
        layerUniformDescriptorSetLayout, maxUBOCountPerLayer, "LayerUniformDescriptorSetLayout");
    buildUniformDescriptorSetLayout(
        drawableUniformDescriptorSetLayout, maxUBOCountPerDrawable, "DrawableUniformDescriptorSetLayout");
    buildImageDescriptorSetLayout();
}

void Context::destroyResources() {
    backend.getDevice()->waitIdle();

    for (auto& frame : frameResources) {
        frame.runDeletionQueue(*this);
    }

    globalUniformBuffers.freeDescriptorSets();

    // all resources have unique handles
    frameResources.clear();
}

void Context::enqueueDeletion(std::optional<std::size_t> threadIndex, std::function<void(Context&)>&& function) {
    if (frameResources.empty()) {
        function(*this);
        return;
    }

    if (threadIndex) {
        frameResources[frameResourceIndex].deletionQueue[*threadIndex+1].push(std::move(function));
    } else {
        std::lock_guard lock(frameResources[frameResourceIndex].deletionQueueMutex);
        frameResources[frameResourceIndex].deletionQueue[0].push(std::move(function));
    }
}

void Context::submitOneTimeCommand(const std::function<void(const vk::UniqueCommandBuffer&)>& function) const {
    MLN_TRACE_FUNC();

    const vk::CommandBufferAllocateInfo allocateInfo(
        backend.getCommandPool().get(), vk::CommandBufferLevel::ePrimary, 1);

    const auto& device = backend.getDevice();
    const auto& commandBuffers = device->allocateCommandBuffersUnique(allocateInfo);
    auto& commandBuffer = commandBuffers.front();

    backend.setDebugName(commandBuffer.get(), "OneTimeSubmitCommandBuffer");

    commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    function(commandBuffer);
    commandBuffer->end();

    const auto submitInfo = vk::SubmitInfo().setCommandBuffers(commandBuffer.get());

    const auto& fence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlags()));
    backend.getGraphicsQueue().submit(submitInfo, fence.get());

    constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();
    const vk::Result waitFenceResult = device->waitForFences(1, &fence.get(), VK_TRUE, timeout);
    if (waitFenceResult != vk::Result::eSuccess) {
        mbgl::Log::Error(mbgl::Event::Render, "OneTimeCommand - Wait fence failed");
    }
}

void Context::waitFrame() const {
    MLN_TRACE_FUNC();
    const auto& device = backend.getDevice();
    auto& frame = frameResources[frameResourceIndex];
    constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();

    const vk::Result waitFenceResult = device->waitForFences(1, &frame.flightFrameFence.get(), VK_TRUE, timeout);
    if (waitFenceResult != vk::Result::eSuccess) {
        mbgl::Log::Error(mbgl::Event::Render, "Wait fence failed");
    }
}
void Context::beginFrame() {
    MLN_TRACE_FUNC();

    const auto& device = backend.getDevice();
    auto& renderableResource = backend.getDefaultRenderable().getResource<SurfaceRenderableResource>();
    const auto& platformSurface = renderableResource.getPlatformSurface();

    if (platformSurface && surfaceUpdateRequested) {
        renderableResource.recreateSwapchain();

        // we wait for an idle device to recreate the swapchain
        // so it's a good opportunity to delete all queued items
        for (auto& frame : frameResources) {
            frame.runDeletionQueue(*this);
        }

        // sync resources with swapchain
        frameResourceIndex = 0;
        surfaceUpdateRequested = false;
    }

    backend.startFrameCapture();

    auto& frame = frameResources[frameResourceIndex];

    waitFrame();

    {
        frame.primaryCommandBuffer->reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        frame.primaryCommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

        frame.uploadCommandBuffer->reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        frame.uploadCommandBuffer->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        for (auto& buffer : frame.secondaryCommandBuffers) {
            if (buffer) {
                buffer->reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            }
        }
    }

    frame.runDeletionQueue(*this);

    if (platformSurface) {
        MLN_TRACE_ZONE(acquireNextImageKHR);
        try {
            constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();
            const vk::ResultValue acquireImageResult = device->acquireNextImageKHR(
                renderableResource.getSwapchain().get(), timeout, frame.surfaceSemaphore.get(), nullptr);

            if (acquireImageResult.result == vk::Result::eSuccess) {
                renderableResource.setAcquiredImageIndex(acquireImageResult.value);
            } else if (acquireImageResult.result == vk::Result::eSuboptimalKHR) {
                renderableResource.setAcquiredImageIndex(acquireImageResult.value);
                // TODO implement pre-rotation transform for surface orientation
#if defined(__APPLE__)
                requestSurfaceUpdate();
                beginFrame();
                return;
#endif
            }

        } catch (const vk::OutOfDateKHRError& e) {
            // request an update and restart frame
            requestSurfaceUpdate();
            beginFrame();
            return;
        }
    } else {
        renderableResource.setAcquiredImageIndex(frameResourceIndex);
    }

    backend.getThreadPool().runRenderJobs();

    // Ensure that everything which will be run on threads has allocated space for
    // per-thread structures so that no locking is necessary during rendering.
    globalUniformBuffers.init(*this, renderThreadCount);
    getGeneralPipelineLayout();
    getPushConstantPipelineLayout();
}

void Context::endFrame() {
    frameResourceIndex = (frameResourceIndex + 1) % frameResources.size();
}

void Context::submitFrame() {
    MLN_TRACE_FUNC();
    const auto& device = backend.getDevice();
    const auto& graphicsQueue = backend.getGraphicsQueue();
    auto& renderableResource = backend.getDefaultRenderable().getResource<SurfaceRenderableResource>();
    const auto& platformSurface = renderableResource.getPlatformSurface();
    const auto& frame = frameResources[frameResourceIndex];

    frame.uploadCommandBuffer->end();
    frame.primaryCommandBuffer->end();

    const vk::Result resetFenceResult = device->resetFences(1, &frame.flightFrameFence.get());
    if (resetFenceResult != vk::Result::eSuccess) {
        mbgl::Log::Error(mbgl::Event::Render, "Reset fence failed");
    }

    if (platformSurface) {
        // submit frame commands
        const vk::CommandBuffer buffers[] = {frame.uploadCommandBuffer.get(), frame.primaryCommandBuffer.get()};
        const vk::PipelineStageFlags waitStageMask[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        auto submitInfo = vk::SubmitInfo()
                              .setCommandBuffers(buffers)
                              .setSignalSemaphores(frame.frameSemaphore.get())
                              .setWaitSemaphores(frame.surfaceSemaphore.get())
                              .setWaitDstStageMask(waitStageMask);
        graphicsQueue.submit(submitInfo, frame.flightFrameFence.get());

        // present rendered frame
        const auto acquiredImage = renderableResource.getAcquiredImageIndex();
        const auto presentInfo = vk::PresentInfoKHR()
                                     .setSwapchains(renderableResource.getSwapchain().get())
                                     .setWaitSemaphores(frame.frameSemaphore.get())
                                     .setImageIndices(acquiredImage);

        try {
            const auto& presentQueue = backend.getPresentQueue();
            const vk::Result presentResult = presentQueue.presentKHR(presentInfo);
            if (presentResult == vk::Result::eSuboptimalKHR) {
                // TODO implement pre-rotation transform for surface orientation
#if defined(__APPLE__)
                requestSurfaceUpdate();
#endif
            }
        } catch (const vk::OutOfDateKHRError& e) {
            requestSurfaceUpdate();
        }
    }

    backend.endFrameCapture();
}

std::unique_ptr<gfx::CommandEncoder> Context::createCommandEncoder() {
    return std::make_unique<CommandEncoder>(*this);
}

std::size_t Context::getThreadIndex(std::int32_t layerIndex, std::int32_t maxLayerIndex) const {
    return (layerIndex * renderThreadCount) / (maxLayerIndex + 1);
}

const vk::UniqueCommandBuffer& Context::getSecondaryCommandBuffer(std::size_t threadIndex) {
    MLN_TRACE_FUNC();
    auto& frame = frameResources[frameResourceIndex];

    assert(0 <= threadIndex && threadIndex < renderThreadCount);
    auto& buffer = frame.secondaryCommandBuffers[threadIndex];

    // Begin the secondary buffer, if we haven't already done so this frame
    if (!frame.secondaryCommandBufferBegin[threadIndex]) {
        const auto& renderableResource = backend.getDefaultRenderable().getResource<SurfaceRenderableResource>();
        const auto& renderPass = renderableResource.getRenderPass();
        const auto& frameBuffer = renderableResource.getFramebuffer();
        const auto inheritInfo = vk::CommandBufferInheritanceInfo{renderPass.get(), 0, frameBuffer.get()};
        buffer->reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        buffer->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eRenderPassContinue, &inheritInfo});
        frame.secondaryCommandBufferBegin[threadIndex] = true;
    }
    return buffer;
}

void Context::endEncoding() {
    auto& frame = frameResources[frameResourceIndex];

    // Encode each secondary buffer into the primary
    for (auto threadIndex = 0_uz; threadIndex < renderThreadCount; ++threadIndex) {
        auto& buffer = frame.secondaryCommandBuffers[threadIndex];
        if (buffer && frame.secondaryCommandBufferBegin[threadIndex]) {
            buffer->end();
            frame.primaryCommandBuffer->executeCommands(buffer.get());
            frame.secondaryCommandBufferBegin[threadIndex] = false;
        }
    }

    frame.primaryCommandBuffer->endRenderPass();
}

BufferResource Context::createBuffer(const void* data, std::size_t size, std::uint32_t usage, bool persistent) const {
    return BufferResource(const_cast<Context&>(*this), data, size, usage, persistent);
}

UniqueShaderProgram Context::createProgram(shaders::BuiltIn shaderID,
                                           std::string name,
                                           const std::string_view vertex,
                                           const std::string_view fragment,
                                           const ProgramParameters& programParameters,
                                           const mbgl::unordered_map<std::string, std::string>& additionalDefines) {
    auto program = std::make_unique<ShaderProgram>(
        shaderID, name, vertex, fragment, programParameters, additionalDefines, backend, *observer);
    return program;
}

gfx::UniqueDrawableBuilder Context::createDrawableBuilder(std::string name) {
    return std::make_unique<DrawableBuilder>(std::move(name));
}

gfx::UniformBufferPtr Context::createUniformBuffer(const void* data, std::size_t size, bool persistent) {
    return std::make_shared<UniformBuffer>(createBuffer(data, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, persistent));
}

gfx::ShaderProgramBasePtr Context::getGenericShader(gfx::ShaderRegistry& shaders, const std::string& name) {
    const auto shaderGroup = shaders.getShaderGroup(name);
    auto shader = shaderGroup ? shaderGroup->getOrCreateShader(*this, {}) : gfx::ShaderProgramBasePtr{};
    return std::static_pointer_cast<gfx::ShaderProgramBase>(std::move(shader));
}

TileLayerGroupPtr Context::createTileLayerGroup(int32_t layerIndex, std::size_t initialCapacity, std::string name) {
    return std::make_shared<TileLayerGroup>(layerIndex, initialCapacity, std::move(name));
}

LayerGroupPtr Context::createLayerGroup(int32_t layerIndex, std::size_t initialCapacity, std::string name) {
    return std::make_shared<LayerGroup>(layerIndex, initialCapacity, name);
}

bool Context::emplaceOrUpdateUniformBuffer(gfx::UniformBufferPtr& buffer,
                                           const void* data,
                                           std::size_t size,
                                           bool persistent) {
    if (buffer) {
        buffer->update(data, size);
        return false;
    } else {
        buffer = createUniformBuffer(data, size, persistent);
        return true;
    }
}

gfx::Texture2DPtr Context::createTexture2D() {
    return std::make_shared<Texture2D>(*this);
}

RenderTargetPtr Context::createRenderTarget(const Size size, const gfx::TextureChannelDataType type) {
    return std::make_shared<RenderTarget>(*this, size, type);
}

std::unique_ptr<gfx::OffscreenTexture> Context::createOffscreenTexture(Size size,
                                                                       gfx::TextureChannelDataType type,
                                                                       bool depth,
                                                                       bool stencil) {
    return std::make_unique<OffscreenTexture>(*this, size, type, depth, stencil);
}

std::unique_ptr<gfx::OffscreenTexture> Context::createOffscreenTexture(Size size, gfx::TextureChannelDataType type) {
    return createOffscreenTexture(size, type, false, false);
}

std::unique_ptr<gfx::TextureResource> Context::createTextureResource(Size,
                                                                     gfx::TexturePixelType,
                                                                     gfx::TextureChannelDataType) {
    throw std::runtime_error("Vulkan TextureResource not implemented");
    return nullptr;
}

std::unique_ptr<gfx::RenderbufferResource> Context::createRenderbufferResource(gfx::RenderbufferPixelType, Size) {
    return std::make_unique<RenderbufferResource>();
}

std::unique_ptr<gfx::DrawScopeResource> Context::createDrawScopeResource() {
    throw std::runtime_error("Vulkan DrawScopeResource not implemented");
    return nullptr;
}

gfx::VertexAttributeArrayPtr Context::createVertexAttributeArray() const {
    return std::make_shared<VertexAttributeArray>();
}

#if !defined(NDEBUG)
void Context::visualizeStencilBuffer() {}

void Context::visualizeDepthBuffer(float) {}
#endif // !defined(NDEBUG)

void Context::clearStencilBuffer(int32_t) {
    // See `PaintParameters::clearStencil`
    assert(false);
}

void Context::bindGlobalUniformBuffers(gfx::RenderPass& renderPass, std::optional<std::size_t> threadIndex) {
    auto& renderPassImpl = static_cast<RenderPass&>(renderPass);
    globalUniformBuffers.bindDescriptorSets(renderPassImpl.getEncoder(), threadIndex);
}

bool Context::renderTileClippingMasks(std::optional<std::size_t> threadIndex,
                                      gfx::RenderPass& renderPass,
                                      RenderStaticData& staticData,
                                      const std::vector<shaders::ClipUBO>& tileUBOs) {
    using ShaderClass = shaders::ShaderSource<shaders::BuiltIn::ClippingMaskProgram, gfx::Backend::Type::Vulkan>;

    std::unique_lock lock{clippingMutex};

    if (!clipping.shader) {
        const auto group = staticData.shaders->getShaderGroup("ClippingMaskProgram");
        if (group) {
            clipping.shader = std::static_pointer_cast<gfx::ShaderProgramBase>(group->getOrCreateShader(*this, {}));
        }
    }
    if (!clipping.shader) {
        assert(!"Failed to create shader for clip masking");
        return false;
    }

    // Create a vertex buffer from the fixed tile coordinates
    if (!clipping.vertexBuffer) {
        const auto vertices = RenderStaticData::tileVertices();
        clipping.vertexBuffer.emplace(
            createBuffer(vertices.data(), vertices.bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false));
    }

    // Create a buffer from the fixed tile indexes
    if (!clipping.indexBuffer) {
        const auto indices = RenderStaticData::quadTriangleIndices();
        clipping.indexBuffer.emplace(
            createBuffer(indices.data(), indices.bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, false));
        clipping.indexCount = 6;
    }

    // build pipeline
    if (clipping.pipelineInfo.inputAttributes.empty()) {
        clipping.pipelineInfo.usePushConstants = true;

        clipping.pipelineInfo.colorBlend = false;
        clipping.pipelineInfo.colorMask = vk::ColorComponentFlags();

        clipping.pipelineInfo.depthTest = false;
        clipping.pipelineInfo.depthWrite = false;

        clipping.pipelineInfo.stencilTest = true;
        clipping.pipelineInfo.stencilFunction = vk::CompareOp::eAlways;
        clipping.pipelineInfo.stencilPass = vk::StencilOp::eReplace;
        clipping.pipelineInfo.dynamicValues.stencilWriteMask = 0b11111111;
        clipping.pipelineInfo.dynamicValues.stencilRef = 0b11111111;

        clipping.pipelineInfo.inputBindings.push_back(
            vk::VertexInputBindingDescription()
                .setBinding(0)
                .setStride(static_cast<uint32_t>(VertexAttribute::getStrideOf(ShaderClass::attributes[0].dataType)))
                .setInputRate(vk::VertexInputRate::eVertex));

        clipping.pipelineInfo.inputAttributes.push_back(
            vk::VertexInputAttributeDescription()
                .setBinding(0)
                .setLocation(static_cast<uint32_t>(ShaderClass::attributes[0].index))
                .setFormat(PipelineInfo::vulkanFormat(ShaderClass::attributes[0].dataType)));
    }

    auto& shaderImpl = static_cast<ShaderProgram&>(*clipping.shader);
    auto& renderPassImpl = static_cast<RenderPass&>(renderPass);
    auto& commandBuffer = renderPassImpl.getEncoder().getCommandBuffer(threadIndex);

    clipping.pipelineInfo.setRenderable(renderPassImpl.getDescriptor().renderable);

    const auto& pipeline = shaderImpl.getPipeline(clipping.pipelineInfo, threadIndex);

    commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
    clipping.pipelineInfo.setDynamicValues(backend, commandBuffer);

    lock.unlock();

    assert(clipping.vertexBuffer->getVulkanBuffer());
    const std::array<vk::Buffer, 1> vertexBuffers = {clipping.vertexBuffer->getVulkanBuffer()};
    const std::array<vk::DeviceSize, 1> offset = {0};

    commandBuffer->bindVertexBuffers(0, vertexBuffers, offset);
    commandBuffer->bindIndexBuffer(clipping.indexBuffer->getVulkanBuffer(), 0, vk::IndexType::eUint16);

    for (const auto& tileInfo : tileUBOs) {
        commandBuffer->setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, tileInfo.stencil_ref);

        commandBuffer->pushConstants(
            getPushConstantPipelineLayout().get(),
            vk::ShaderStageFlags() | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            sizeof(tileInfo.matrix),
            &tileInfo.matrix);
        commandBuffer->drawIndexed(clipping.indexCount, 1, 0, 0, 0);
    }

    stats.numDrawCalls++;
    stats.totalDrawCalls++;
    return true;
}

const std::unique_ptr<BufferResource>& Context::getDummyVertexBuffer() {
    if (!dummyVertexBuffer)
        dummyVertexBuffer = std::make_unique<BufferResource>(
            *this, nullptr, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
    return dummyVertexBuffer;
}

const std::unique_ptr<BufferResource>& Context::getDummyUniformBuffer() {
    if (!dummyUniformBuffer)
        dummyUniformBuffer = std::make_unique<BufferResource>(
            *this, nullptr, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
    return dummyUniformBuffer;
}

const std::unique_ptr<Texture2D>& Context::getDummyTexture(std::optional<std::size_t> threadIndex) {
    if (!dummyTexture2D) {
        const Size size(2, 2);
        const std::vector<Color> data(4ull * size.width * size.height, Color::white());

        dummyTexture2D = std::make_unique<Texture2D>(*this);
        dummyTexture2D->setFormat(gfx::TexturePixelType::RGBA, gfx::TextureChannelDataType::UnsignedByte);
        dummyTexture2D->setSize(size);

        submitOneTimeCommand([&](const vk::UniqueCommandBuffer& commandBuffer) {
            dummyTexture2D->uploadSubRegion(data.data(), size, 0, 0, commandBuffer, threadIndex);
        });
    }

    return dummyTexture2D;
}

void Context::buildUniformDescriptorSetLayout(vk::UniqueDescriptorSetLayout& layout,
                                              size_t uniformCount,
                                              const std::string& name) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    const auto stageFlags = vk::ShaderStageFlags() | vk::ShaderStageFlagBits::eVertex |
                            vk::ShaderStageFlagBits::eFragment;

    for (size_t i = 0; i < uniformCount; ++i) {
        bindings.push_back(vk::DescriptorSetLayoutBinding()
                               .setBinding(static_cast<uint32_t>(i))
                               .setStageFlags(stageFlags)
                               .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                               .setDescriptorCount(1));
    }

    const auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(bindings);
    layout = backend.getDevice()->createDescriptorSetLayoutUnique(descriptorSetLayoutCreateInfo);
    backend.setDebugName(layout.get(), name);
}

void Context::buildImageDescriptorSetLayout() {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

    for (size_t i = 0; i < shaders::maxTextureCountPerShader; ++i) {
        bindings.push_back(vk::DescriptorSetLayoutBinding()
                               .setBinding(static_cast<uint32_t>(i))
                               .setStageFlags(vk::ShaderStageFlagBits::eFragment)
                               .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                               .setDescriptorCount(1));
    }

    const auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(bindings);
    drawableImageDescriptorSetLayout = backend.getDevice()->createDescriptorSetLayoutUnique(
        descriptorSetLayoutCreateInfo);
    backend.setDebugName(drawableImageDescriptorSetLayout.get(), "ImageDescriptorSetLayout");
}

const vk::DescriptorSetLayout& Context::getDescriptorSetLayout(DescriptorSetType type) {
    switch (type) {
        case DescriptorSetType::Global:
            return globalUniformDescriptorSetLayout.get();

        case DescriptorSetType::Layer:
            return layerUniformDescriptorSetLayout.get();

        case DescriptorSetType::DrawableUniform:
            return drawableUniformDescriptorSetLayout.get();

        case DescriptorSetType::DrawableImage:
            return drawableImageDescriptorSetLayout.get();

        default:
            assert(static_cast<uint32_t>(type) < static_cast<uint32_t>(DescriptorSetType::Count));
            return globalUniformDescriptorSetLayout.get();
            break;
    }
}

DescriptorPoolGrowable& Context::getDescriptorPool(DescriptorSetType type, std::optional<std::size_t> threadIndex) {
    assert(static_cast<uint32_t>(type) < static_cast<uint32_t>(DescriptorSetType::Count));
    return descriptorPoolMaps[threadIndex ? *threadIndex + 1 : 0][type];
}

const vk::UniquePipelineLayout& Context::getGeneralPipelineLayout() {
    if (generalPipelineLayout) {
        return generalPipelineLayout;
    }

    const std::vector<vk::DescriptorSetLayout> layouts = {
        globalUniformDescriptorSetLayout.get(),
        layerUniformDescriptorSetLayout.get(),
        drawableUniformDescriptorSetLayout.get(),
        drawableImageDescriptorSetLayout.get(),
    };

    generalPipelineLayout = backend.getDevice()->createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo().setSetLayouts(layouts));

    backend.setDebugName(generalPipelineLayout.get(), "PipelineLayout_general");

    return generalPipelineLayout;
}

const vk::UniquePipelineLayout& Context::getPushConstantPipelineLayout() {
    if (pushConstantPipelineLayout) {
        return pushConstantPipelineLayout;
    }

    const auto stages = vk::ShaderStageFlags() | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    const auto pushConstant = vk::PushConstantRange().setSize(sizeof(matf4)).setStageFlags(stages);

    pushConstantPipelineLayout = backend.getDevice()->createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo().setPushConstantRanges(pushConstant));

    backend.setDebugName(pushConstantPipelineLayout.get(), "PipelineLayout_pushConstants");

    return pushConstantPipelineLayout;
}

void Context::FrameResources::runDeletionQueue(Context& context) {
    MLN_TRACE_FUNC();

    std::lock_guard lock(deletionQueueMutex);

    for (auto& threadQueue : deletionQueue) {
        while (!threadQueue.empty()) {
            threadQueue.front()(context);
            threadQueue.pop();
        }
    }
}

} // namespace vulkan
} // namespace mbgl
