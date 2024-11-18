#pragma once

#include <mbgl/gfx/texture.hpp>
#include <mbgl/gfx/draw_mode.hpp>
#include <mbgl/gfx/depth_mode.hpp>
#include <mbgl/gfx/stencil_mode.hpp>
#include <mbgl/gfx/color_mode.hpp>
#include <mbgl/gfx/texture2d.hpp>
#include <mbgl/gfx/context.hpp>
#include <mbgl/util/containers.hpp>
#include <mbgl/util/noncopyable.hpp>
#include <mbgl/util/std.hpp>
#include <mbgl/vulkan/uniform_buffer.hpp>
#include <mbgl/vulkan/renderer_backend.hpp>
#include <mbgl/vulkan/pipeline.hpp>
#include <mbgl/vulkan/descriptor_set.hpp>

#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace mbgl {

class ProgramParameters;
class RenderStaticData;

namespace gfx {
class VertexAttributeArray;
using VertexAttributeArrayPtr = std::shared_ptr<VertexAttributeArray>;
} // namespace gfx

namespace shaders {
struct ClipUBO;
} // namespace shaders

namespace vulkan {

class RenderPass;
class RendererBackend;
class ShaderProgram;
class VertexBufferResource;
class Texture2D;

using UniqueShaderProgram = std::unique_ptr<ShaderProgram>;
using UniqueVertexBufferResource = std::unique_ptr<VertexBufferResource>;

class Context final : public gfx::Context {
public:
    Context(RendererBackend&);
    ~Context() noexcept override;
    Context(const Context&) = delete;
    Context& operator=(const Context& other) = delete;

    RendererBackend& getBackend() const { return backend; }

    auto getRenderThreadCount() const { return renderThreadCount; }
    std::size_t getThreadIndex(std::int32_t layerIndex, std::int32_t maxLayerIndex) const;

    void beginFrame() override;
    void endFrame() override;
    void submitFrame();
    void waitFrame() const;

    std::unique_ptr<gfx::CommandEncoder> createCommandEncoder() override;

    BufferResource createBuffer(const void* data, std::size_t size, std::uint32_t usage, bool persistent) const;

    UniqueShaderProgram createProgram(shaders::BuiltIn shaderID,
                                      std::string name,
                                      const std::string_view vertex,
                                      const std::string_view fragment,
                                      const ProgramParameters& programParameters,
                                      const mbgl::unordered_map<std::string, std::string>& additionalDefines);

    /// Called at the end of a frame.
    void performCleanup() override {}
    void reduceMemoryUsage() override {}

    gfx::UniqueDrawableBuilder createDrawableBuilder(std::string name) override;
    gfx::UniformBufferPtr createUniformBuffer(const void* data, std::size_t size, bool persistent) override;

    gfx::ShaderProgramBasePtr getGenericShader(gfx::ShaderRegistry&, const std::string& name) override;

    TileLayerGroupPtr createTileLayerGroup(int32_t layerIndex, std::size_t initialCapacity, std::string name) override;

    LayerGroupPtr createLayerGroup(int32_t layerIndex, std::size_t initialCapacity, std::string name) override;

    gfx::Texture2DPtr createTexture2D() override;

    RenderTargetPtr createRenderTarget(const Size size, const gfx::TextureChannelDataType type) override;

    void resetState(gfx::DepthMode, gfx::ColorMode) override {}

    bool emplaceOrUpdateUniformBuffer(gfx::UniformBufferPtr&,
                                      const void* data,
                                      std::size_t size,
                                      bool persistent = false) override;

    void setDirtyState() override {}

    std::unique_ptr<gfx::OffscreenTexture> createOffscreenTexture(Size, gfx::TextureChannelDataType, bool, bool);

    std::unique_ptr<gfx::OffscreenTexture> createOffscreenTexture(Size, gfx::TextureChannelDataType) override;

    std::unique_ptr<gfx::TextureResource> createTextureResource(Size,
                                                                gfx::TexturePixelType,
                                                                gfx::TextureChannelDataType) override;

    std::unique_ptr<gfx::RenderbufferResource> createRenderbufferResource(gfx::RenderbufferPixelType,
                                                                          Size size) override;

    std::unique_ptr<gfx::DrawScopeResource> createDrawScopeResource() override;

    gfx::VertexAttributeArrayPtr createVertexAttributeArray() const override;

#if !defined(NDEBUG)
    void visualizeStencilBuffer() override;
    void visualizeDepthBuffer(float depthRangeSize) override;
#endif

    void clearStencilBuffer(int32_t) override;

    /// Get the global uniform buffers
    const gfx::UniformBufferArray& getGlobalUniformBuffers() const override { return globalUniformBuffers; };

    /// Get the mutable global uniform buffer array
    gfx::UniformBufferArray& mutableGlobalUniformBuffers() override { return globalUniformBuffers; };

    /// Bind the global uniform buffers
    void bindGlobalUniformBuffers(gfx::RenderPass&, std::optional<std::size_t> threadIndex) override;

    /// Unbind the global uniform buffers
    void unbindGlobalUniformBuffers(gfx::RenderPass&, std::optional<std::size_t>) override {}

    bool renderTileClippingMasks(std::optional<std::size_t> threadIndex,
                                 gfx::RenderPass& renderPass,
                                 RenderStaticData& staticData,
                                 const std::vector<shaders::ClipUBO>& tileUBOs);

    const std::unique_ptr<BufferResource>& getDummyVertexBuffer();
    const std::unique_ptr<BufferResource>& getDummyUniformBuffer();
    const std::unique_ptr<Texture2D>& getDummyTexture(std::optional<std::size_t> threadIndex);

    const vk::DescriptorSetLayout& getDescriptorSetLayout(DescriptorSetType type);
    const vk::UniquePipelineLayout& getGeneralPipelineLayout();
    const vk::UniquePipelineLayout& getPushConstantPipelineLayout();

    DescriptorPoolGrowable& getDescriptorPool(DescriptorSetType, std::optional<std::size_t> threadIndex);

    uint8_t getCurrentFrameResourceIndex() const { return frameResourceIndex; }
    void enqueueDeletion(std::optional<std::size_t> threadIndex, std::function<void(Context&)>&& function);
    void submitOneTimeCommand(const std::function<void(const vk::UniqueCommandBuffer&)>& function) const;

    const vk::UniqueCommandBuffer& getPrimaryCommandBuffer() const {
        return frameResources[frameResourceIndex].primaryCommandBuffer;
    }
    const vk::UniqueCommandBuffer& getUploadCommandBuffer() const {
        return frameResources[frameResourceIndex].uploadCommandBuffer;
    }
    const vk::UniqueCommandBuffer& getSecondaryCommandBuffer(std::size_t threadIndex);

    const vk::UniqueCommandBuffer& getCommandBuffer(std::optional<std::size_t> threadIndex) {
        if (!renderThreadCount) {
            return frameResources[frameResourceIndex].primaryCommandBuffer;
        } else if (!threadIndex) {
            return frameResources[frameResourceIndex].uploadCommandBuffer;
        }
        return getSecondaryCommandBuffer(*threadIndex);
    }

    void endEncoding();

    void requestSurfaceUpdate() { surfaceUpdateRequested = true; }

private:
    template <typename Func>
        requires requires(Func f, std::size_t threadIndex) {
            { f(threadIndex) } -> std::same_as<void>;
        }
    void eachRenderThread(Func f) {
        for (auto i = 0_uz; i < renderThreadCount; ++i) {
            f(i);
        }
    }

    struct FrameResources {
        vk::UniqueCommandBuffer primaryCommandBuffer;
        vk::UniqueCommandBuffer uploadCommandBuffer;

        std::vector<vk::UniqueCommandBuffer> secondaryCommandBuffers;
        std::vector<bool> secondaryCommandBufferBegin;

        vk::UniqueSemaphore surfaceSemaphore;
        vk::UniqueSemaphore frameSemaphore;
        vk::UniqueFence flightFrameFence;

        std::vector<std::queue<std::function<void(Context&)>>> deletionQueue;
        MLN_TRACE_LOCKABLE(std::mutex, deletionQueueMutex); // protects only element 0

        explicit FrameResources(std::size_t threadCount,
                                vk::UniqueCommandBuffer& pcb,
                                vk::UniqueCommandBuffer& ucb,
                                vk::UniqueSemaphore&& surf,
                                vk::UniqueSemaphore&& frame,
                                vk::UniqueFence&& flight)
            : primaryCommandBuffer(std::move(pcb)),
              uploadCommandBuffer(std::move(ucb)),
              secondaryCommandBuffers(threadCount),
              secondaryCommandBufferBegin(threadCount),
              surfaceSemaphore(std::move(surf)),
              frameSemaphore(std::move(frame)),
              flightFrameFence(std::move(flight)),
              deletionQueue(threadCount + 1){}
        FrameResources(FrameResources&& other) :
         primaryCommandBuffer(std::move(other.primaryCommandBuffer)),
         uploadCommandBuffer(std::move(other.uploadCommandBuffer)),
         secondaryCommandBuffers(std::move(other.secondaryCommandBuffers)),
         secondaryCommandBufferBegin(std::move(other.secondaryCommandBufferBegin)),
         surfaceSemaphore(std::move(other.surfaceSemaphore)),
         frameSemaphore(std::move(other.frameSemaphore)),
         flightFrameFence(std::move(other.flightFrameFence)),
         deletionQueue(std::move(other.deletionQueue))
        // not `deletionQueueMutex`
        {}
        FrameResources(const FrameResources&) = delete;

        void runDeletionQueue(Context&);
    };

    void initFrameResources();
    void destroyResources();

    void buildImageDescriptorSetLayout();
    void buildUniformDescriptorSetLayout(vk::UniqueDescriptorSetLayout& layout,
                                         size_t uniformCount,
                                         const std::string& name);

private:
    RendererBackend& backend;
    const std::size_t renderThreadCount;

    vulkan::UniformBufferArray globalUniformBuffers;

    using DescriptorPoolMap = std::unordered_map<DescriptorSetType, DescriptorPoolGrowable>;
    std::vector<DescriptorPoolMap> descriptorPoolMaps;

    std::unique_ptr<BufferResource> dummyVertexBuffer;
    std::unique_ptr<BufferResource> dummyUniformBuffer;
    std::unique_ptr<Texture2D> dummyTexture2D;
    vk::UniqueDescriptorSetLayout globalUniformDescriptorSetLayout;
    vk::UniqueDescriptorSetLayout layerUniformDescriptorSetLayout;
    vk::UniqueDescriptorSetLayout drawableUniformDescriptorSetLayout;
    vk::UniqueDescriptorSetLayout drawableImageDescriptorSetLayout;
    vk::UniquePipelineLayout generalPipelineLayout;
    vk::UniquePipelineLayout pushConstantPipelineLayout;

    uint8_t frameResourceIndex = 0;
    std::vector<FrameResources> frameResources;
    bool surfaceUpdateRequested{false};

    struct {
        gfx::ShaderProgramBasePtr shader;
        std::optional<BufferResource> vertexBuffer;
        std::optional<BufferResource> indexBuffer;
        uint32_t indexCount = 0;

        PipelineInfo pipelineInfo;
    } clipping;
    MLN_TRACE_LOCKABLE(std::recursive_mutex, clippingMutex);
};

} // namespace vulkan
} // namespace mbgl
