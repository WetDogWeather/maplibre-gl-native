#include <mbgl/vulkan/descriptor_set.hpp>

#include <mbgl/vulkan/command_encoder.hpp>
#include <mbgl/vulkan/context.hpp>
#include <mbgl/vulkan/texture2d.hpp>
#include <mbgl/util/instrumentation.hpp>
#include <mbgl/util/logging.hpp>
#include <mbgl/util/std.hpp>
#include <mbgl/util/traits.hpp>

#include <cassert>
#include <cmath>
#include <ranges>

#define USE_DESCRIPTOR_POOL_RESET

namespace mbgl {
namespace vulkan {

DescriptorSet::DescriptorSet(Context& context_, DescriptorSetType type_, std::size_t threadCount_)
    : context(context_),
      type(type_),
      threadCount(threadCount_),
      threads(threadCount + 1) {}

DescriptorSet::~DescriptorSet() {
    context.enqueueDeletion([type_ = type, threads_ = std::move(threads)](auto& context_) mutable {
        [[maybe_unused]] auto& device = context_.getBackend().getDevice();
        for (auto i = 0_uz; i < threads_.size(); ++i) {
            const auto& thread = threads_[i];
            if (const auto poolIndex = thread.descriptorPoolIndex; 0 <= poolIndex) {
                const auto threadIndex = i ? std::optional<std::size_t>(i - 1) : std::nullopt;
                auto& poolInfo = context_.getDescriptorPool(type_, threadIndex).pools[poolIndex];

#ifdef USE_DESCRIPTOR_POOL_RESET
                poolInfo.unusedSets.push(std::move(thread.descriptorSets));
#else
                device->freeDescriptorSets(poolInfo.pool.get(), thread.descriptorSets);
                poolInfo.remainingSets += thread.descriptorSets.size();
#endif
            }
        }
    });
}

void DescriptorSet::createDescriptorPool(DescriptorPoolGrowable& growablePool) {
    MLN_TRACE_FUNC();
    const auto& device = context.getBackend().getDevice();

    const uint32_t maxSets = static_cast<uint32_t>(growablePool.maxSets *
                                                   std::pow(growablePool.growFactor, growablePool.pools.size()));
    const vk::DescriptorPoolSize size = {type != DescriptorSetType::DrawableImage
                                             ? vk::DescriptorType::eUniformBuffer
                                             : vk::DescriptorType::eCombinedImageSampler,
                                         maxSets * growablePool.descriptorsPerSet};

#ifdef USE_DESCRIPTOR_POOL_RESET
    const auto poolFlags = vk::DescriptorPoolCreateFlags();
#else
    const auto poolFlags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
#endif

    const auto descriptorPoolInfo = vk::DescriptorPoolCreateInfo(poolFlags).setPoolSizes(size).setMaxSets(maxSets);

    {
        MLN_TRACE_ZONE(createDescriptorPoolUnique);
        growablePool.pools.emplace_back(device->createDescriptorPoolUnique(descriptorPoolInfo), maxSets);
        growablePool.currentPoolIndex = static_cast<std::int32_t>(growablePool.pools.size() - 1);
    }
}

void DescriptorSet::allocate(std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();

    auto& thread = threads[indexFor(threadIndex)];
    if (!thread.descriptorSets.empty()) {
        return;
    }

    const auto& device = context.getBackend().getDevice();
    const auto& descriptorSetLayout = context.getDescriptorSetLayout(type);
    auto& growablePool = context.getDescriptorPool(type, threadIndex);
    const std::vector<vk::DescriptorSetLayout> layouts(context.getBackend().getMaxFrames(), descriptorSetLayout);

    if (growablePool.currentPoolIndex == -1 ||
        (growablePool.current().unusedSets.empty() && growablePool.current().remainingSets < layouts.size())) {
#ifdef USE_DESCRIPTOR_POOL_RESET
        // find a pool that has unused allocated descriptor sets
        const auto& unusedPoolIt = std::find_if(
            growablePool.pools.begin(), growablePool.pools.end(), [&](const auto& p) { return !p.unusedSets.empty(); });

        if (unusedPoolIt != growablePool.pools.end()) {
            growablePool.currentPoolIndex = std::distance(growablePool.pools.begin(), unusedPoolIt);
        } else
#endif
        {
            // find a pool that has available memory to allocate more descriptor sets
            const auto& freePoolIt = std::find_if(growablePool.pools.begin(),
                                                  growablePool.pools.end(),
                                                  [&](const auto& p) { return p.remainingSets >= layouts.size(); });

            if (freePoolIt != growablePool.pools.end()) {
                growablePool.currentPoolIndex = std::distance(growablePool.pools.begin(), freePoolIt);
            } else {
                createDescriptorPool(growablePool);
            }
        }
    }

    thread.descriptorPoolIndex = growablePool.currentPoolIndex;

#ifdef USE_DESCRIPTOR_POOL_RESET
    if (!growablePool.current().unusedSets.empty()) {
        thread.descriptorSets = growablePool.current().unusedSets.front();
        growablePool.current().unusedSets.pop();
    } else
#endif
    {
        thread.descriptorSets = device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo()
                                                            .setDescriptorPool(growablePool.current().pool.get())
                                                            .setSetLayouts(layouts));
        growablePool.current().remainingSets -= thread.descriptorSets.size();
    }

    markDirty(threadIndex, true);
}

void DescriptorSet::markDirty(std::optional<std::size_t> threadIndex, bool value) {
    auto& thread = threads[indexFor(threadIndex)];
    thread.dirty.resize(std::max(thread.dirty.size(), thread.descriptorSets.size()));
    std::fill(thread.dirty.begin(), thread.dirty.end(), value);
}

void DescriptorSet::markDirty(AllThreadsTag, bool value) {
    for (auto i = 0_uz; i <= threadCount; ++i) {
        markDirty(i ? i - 1 : std::optional<std::size_t>{}, value);
    }
}

void DescriptorSet::bind(CommandEncoder&, std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();
    auto& commandBuffer = context.getCommandBuffer(threadIndex);
    const uint8_t frameIndex = context.getCurrentFrameResourceIndex();
    auto& thread = threads[indexFor(threadIndex)];
    commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      context.getGeneralPipelineLayout().get(),
                                      static_cast<uint32_t>(type),
                                      thread.descriptorSets[frameIndex],
                                      nullptr);
}

UniformDescriptorSet::UniformDescriptorSet(Context& context_, DescriptorSetType type_, std::size_t threadCount_)
    : DescriptorSet(context_, type_, threadCount_) {}

void UniformDescriptorSet::update(const gfx::UniformBufferArray& uniforms,
                                  uint32_t uniformStartIndex,
                                  uint32_t descriptorBindingCount,
                                  std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();
    allocate(threadIndex);

    auto& thread = threads[indexFor(threadIndex)];
    const uint8_t frameIndex = context.getCurrentFrameResourceIndex();

    thread.dirty.resize(std::max(thread.dirty.size(), thread.descriptorSets.size()));
    if (!thread.dirty[frameIndex]) {
        return;
    }
    thread.dirty[frameIndex] = false;

    const auto& device = context.getBackend().getDevice();

    for (uint32_t index = 0; index < descriptorBindingCount; ++index) {
        MLN_TRACE_ZONE(update);
        MLN_ZONE_VALUE(index);
        vk::DescriptorBufferInfo descriptorBufferInfo;

        if (const auto& uniformBuffer = uniforms.get(uniformStartIndex + index)) {
            MLN_TRACE_ZONE(set);
            const auto& uniformBufferImpl = static_cast<const UniformBuffer&>(*uniformBuffer);
            const auto& bufferResource = uniformBufferImpl.getBufferResource();
            descriptorBufferInfo.setBuffer(bufferResource.getVulkanBuffer())
                .setOffset(bufferResource.getVulkanBufferOffset())
                .setRange(bufferResource.getSizeInBytes());
        } else {
            MLN_TRACE_ZONE(set);
            descriptorBufferInfo.setBuffer(context.getDummyUniformBuffer()->getVulkanBuffer())
                .setOffset(0)
                .setRange(VK_WHOLE_SIZE);
        }

        const auto writeDescriptorSet = vk::WriteDescriptorSet()
                                            .setBufferInfo(descriptorBufferInfo)
                                            .setDescriptorCount(1)
                                            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                                            .setDstBinding(index)
                                            .setDstSet(thread.descriptorSets[frameIndex]);

        {
            MLN_TRACE_ZONE(updateDescriptorSets);
            device->updateDescriptorSets(writeDescriptorSet, nullptr);
        }
    }

    thread.dirty[frameIndex] = false;
}

ImageDescriptorSet::ImageDescriptorSet(Context& context_, std::size_t threadCount_)
    : DescriptorSet(context_, DescriptorSetType::DrawableImage, threadCount_) {}

void ImageDescriptorSet::update(const std::array<gfx::Texture2DPtr, shaders::maxTextureCountPerShader>& textures,
                                std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();
    allocate(threadIndex);

    const uint8_t frameIndex = context.getCurrentFrameResourceIndex();
    auto& thread = threads[indexFor(threadIndex)];

    thread.dirty.resize(std::max(thread.dirty.size(), thread.descriptorSets.size()));
    if (!thread.dirty[frameIndex]) {
        return;
    }
    thread.dirty[frameIndex] = false;

    const auto& device = context.getBackend().getDevice();

    for (size_t id = 0; id < shaders::maxTextureCountPerShader; ++id) {
        const auto& texture = id < textures.size() ? textures[id] : nullptr;
        auto& textureImpl = texture ? static_cast<Texture2D&>(*texture) : *context.getDummyTexture();

        const auto descriptorImageInfo = vk::DescriptorImageInfo()
                                             .setImageLayout(textureImpl.getVulkanImageLayout())
                                             .setImageView(textureImpl.getVulkanImageView().get())
                                             .setSampler(textureImpl.getVulkanSampler());

        const auto writeDescriptorSet = vk::WriteDescriptorSet()
                                            .setImageInfo(descriptorImageInfo)
                                            .setDescriptorCount(1)
                                            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                                            .setDstBinding(static_cast<std::uint32_t>(id))
                                            .setDstSet(thread.descriptorSets[frameIndex]);

        device->updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    thread.dirty[frameIndex] = false;
}

} // namespace vulkan
} // namespace mbgl
