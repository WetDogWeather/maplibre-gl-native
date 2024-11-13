#include <mbgl/vulkan/uniform_buffer.hpp>

#include <mbgl/vulkan/context.hpp>
#include <mbgl/util/logging.hpp>
#include <mbgl/vulkan/command_encoder.hpp>

#include <cassert>

namespace mbgl {
namespace vulkan {

UniformBuffer::UniformBuffer(BufferResource&& buffer_)
    : gfx::UniformBuffer(buffer_.getSizeInBytes()),
      buffer(std::move(buffer_)) {
    buffer.getContext().renderingStats().numUniformBuffers++;
    buffer.getContext().renderingStats().memUniformBuffers += static_cast<int>(size);
}

UniformBuffer::UniformBuffer(UniformBuffer&& other)
    : gfx::UniformBuffer(std::move(other)),
      buffer(std::move(other.buffer)) {}

UniformBuffer::~UniformBuffer() {
    buffer.getContext().renderingStats().numUniformBuffers--;
    buffer.getContext().renderingStats().memUniformBuffers -= static_cast<int>(size);
}

void UniformBuffer::update(const void* data, std::size_t size_) {
    if (size != size_ || size != buffer.getSizeInBytes()) {
        Log::Error(
            Event::General,
            "Mismatched size given to UBO update, expected " + std::to_string(size) + ", got " + std::to_string(size_));
        assert(false);
        return;
    }

    buffer.getContext().renderingStats().numUniformUpdates++;
    buffer.getContext().renderingStats().uniformUpdateBytes += size_;
    buffer.update(data, size, /*offset=*/0);
}

UniformBufferArray::UniformBufferArray(DescriptorSetType descriptorSetType_,
                                       uint32_t descriptorStartIndex_,
                                       uint32_t descriptorBindingCount_)
    : descriptorSetType(descriptorSetType_),
      descriptorStartIndex(descriptorStartIndex_),
      descriptorBindingCount(descriptorBindingCount_) {}

UniformBufferArray::UniformBufferArray(UniformBufferArray&& other)
    : gfx::UniformBufferArray(std::move(other)),
      descriptorSet(std::move(other.descriptorSet)) {}

void UniformBufferArray::init(gfx::Context& context, std::size_t threadCount) {
    if (!descriptorSet) {
        descriptorSet = std::make_unique<UniformDescriptorSet>(
            static_cast<Context&>(context), descriptorSetType, threadCount);
    }
}

const std::shared_ptr<gfx::UniformBuffer>& UniformBufferArray::set(const std::size_t id,
                                                                   std::shared_ptr<gfx::UniformBuffer> uniformBuffer,
                                                                   std::optional<std::size_t> threadIndex) {
    if (id >= uniformBufferVector.size()) {
        return nullref;
    }

    if (uniformBufferVector[id] == uniformBuffer) {
        return uniformBufferVector[id];
    }

    if (descriptorSet) {
        descriptorSet->markDirty(threadIndex);
    }

    uniformBufferVector[id] = std::move(uniformBuffer);
    return uniformBufferVector[id];
}

void UniformBufferArray::createOrUpdate(const size_t id,
                                        const void* data,
                                        std::size_t size,
                                        gfx::Context& context,
                                        std::optional<std::size_t> threadIndex,
                                        bool persistent) {
    if (descriptorSet) {
        if (auto& ubo = get(id); !ubo || ubo->getSize() != size) {
            descriptorSet->markDirty(threadIndex);
        }
    }

    gfx::UniformBufferArray::createOrUpdate(id, data, size, context, threadIndex, persistent);
}

void UniformBufferArray::bindDescriptorSets(CommandEncoder& encoder, std::optional<std::size_t> threadIndex) {
    MLN_TRACE_FUNC();
    assert(descriptorSet);
    {
        MLN_TRACE_ZONE(update);
        descriptorSet->update(*this, descriptorStartIndex, descriptorBindingCount, threadIndex);
    }
    {
        MLN_TRACE_ZONE(bind);
        descriptorSet->bind(encoder, threadIndex);
    }
}

} // namespace vulkan
} // namespace mbgl
