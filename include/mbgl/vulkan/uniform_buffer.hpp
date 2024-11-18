#pragma once

#include <mbgl/gfx/uniform_buffer.hpp>
#include <mbgl/vulkan/buffer_resource.hpp>
#include <mbgl/vulkan/descriptor_set.hpp>

namespace mbgl {
namespace vulkan {

class UniformBuffer final : public gfx::UniformBuffer {
public:
    UniformBuffer(BufferResource&&);
    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer(UniformBuffer&&);
    ~UniformBuffer() override;

    const BufferResource& getBufferResource() const { return buffer; }

    void releaseResource(std::optional<std::size_t> threadIndex);

    UniformBuffer clone() const { return {buffer.clone()}; }

    void update(const void* data, std::size_t size_) override;

protected:
    BufferResource buffer;
};

/// Stores a collection of uniform buffers by name
class UniformBufferArray final : public gfx::UniformBufferArray {
public:
    UniformBufferArray() = delete;
    UniformBufferArray(DescriptorSetType, uint32_t descriptorStartIndex, uint32_t descriptorBindingCount);

    UniformBufferArray(UniformBufferArray&&);
    UniformBufferArray(const UniformBufferArray&) = delete;

    UniformBufferArray& operator=(UniformBufferArray&& other) {
        gfx::UniformBufferArray::operator=(std::move(other));
        return *this;
    }
    UniformBufferArray& operator=(const UniformBufferArray& other) {
        gfx::UniformBufferArray::operator=(other);
        return *this;
    }

    ~UniformBufferArray() = default;

    void init(gfx::Context&, std::size_t threadCount);

    const std::shared_ptr<gfx::UniformBuffer>& set(const size_t id,
                                                   std::shared_ptr<gfx::UniformBuffer> uniformBuffer,
                                                   std::optional<std::size_t> threadIndex) override;

    void createOrUpdate(const size_t id,
                        const void* data,
                        std::size_t size,
                        gfx::Context& context,
                        std::optional<std::size_t> threadIndex,
                        bool persistent) override;

    void bindDescriptorSets(CommandEncoder& encoder, std::optional<std::size_t> threadIndex);
    void freeDescriptorSets() { descriptorSet.reset(); }

private:
    gfx::UniqueUniformBuffer copy(const gfx::UniformBuffer& buffer) override {
        return std::make_unique<UniformBuffer>(static_cast<const UniformBuffer&>(buffer).clone());
    }

    const DescriptorSetType descriptorSetType{DescriptorSetType::DrawableUniform};
    const uint32_t descriptorStartIndex{0};
    const uint32_t descriptorBindingCount{0};

    std::unique_ptr<UniformDescriptorSet> descriptorSet;
};

} // namespace vulkan
} // namespace mbgl
