#pragma once

#include <mbgl/gfx/uniform_buffer.hpp>
#include <mbgl/vulkan/buffer_resource.hpp>
#include <span>
#include <queue>

namespace mbgl {
namespace vulkan {

class CommandEncoder;
class Context;

enum class DescriptorSetType : uint8_t {
    Global,
    Layer,
    DrawableUniform,
    DrawableImage,
    Count,
};

struct DescriptorPoolGrowable {
    struct PoolInfo {
        vk::UniqueDescriptorPool pool;
        uint32_t remainingSets{0};
        std::queue<std::vector<vk::DescriptorSet>> unusedSets;

        PoolInfo(vk::UniqueDescriptorPool&& pool_, uint32_t remainingSets_)
            : pool(std::move(pool_)),
              remainingSets(remainingSets_) {}
    };

    const uint32_t maxSets{0};
    const uint32_t descriptorsPerSet{0};
    const float growFactor{1.5f};

    std::vector<PoolInfo> pools;
    int32_t currentPoolIndex{-1};

    PoolInfo& current() {
        assert(0 <= currentPoolIndex);
        return pools[currentPoolIndex];
    }

    DescriptorPoolGrowable() = default;
    DescriptorPoolGrowable(uint32_t maxSets_, uint32_t descriptorsPerSet_, float growFactor_ = 1.5f)
        : maxSets(maxSets_),
          descriptorsPerSet(descriptorsPerSet_),
          growFactor(growFactor_) {}
};

class DescriptorSet {
public:
    DescriptorSet(Context&, DescriptorSetType, std::size_t threadCount_);
    virtual ~DescriptorSet();

    void allocate(std::optional<std::size_t> threadIndex);

    void markDirty(std::optional<std::size_t> threadIndex, bool value = true);

    struct AllThreadsTag {};
    static constexpr AllThreadsTag AllThreads;
    void markDirty(AllThreadsTag, bool value = true);

    void bind(CommandEncoder&, std::optional<std::size_t> threadIndex);

protected:
    void createDescriptorPool(DescriptorPoolGrowable& growablePool);

    std::size_t indexFor(std::optional<std::size_t> threadIndex) const {
        assert(!threadIndex || *threadIndex < threads.size());
        return threadIndex ? *threadIndex + 1 : 0;
    }

protected:
    Context& context;
    const DescriptorSetType type;
    const std::size_t threadCount;

    struct PerThreadData {
        std::vector<bool> dirty;
        std::vector<vk::DescriptorSet> descriptorSets;
        int32_t descriptorPoolIndex{-1};
    };
    std::vector<PerThreadData> threads;
};

class UniformDescriptorSet : public DescriptorSet {
public:
    UniformDescriptorSet(Context&, DescriptorSetType, std::size_t threadCount);
    virtual ~UniformDescriptorSet() = default;

    void update(const gfx::UniformBufferArray& uniforms,
                uint32_t uniformStartIndex,
                uint32_t descriptorBindingCount,
                std::optional<std::size_t> threadIndex);
};

class ImageDescriptorSet : public DescriptorSet {
public:
    ImageDescriptorSet(Context&, std::size_t threadCount);
    virtual ~ImageDescriptorSet() = default;

    void update(const std::array<gfx::Texture2DPtr, shaders::maxTextureCountPerShader>& textures,
                std::optional<std::size_t> threadIndex);
};

} // namespace vulkan
} // namespace mbgl
