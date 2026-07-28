#include "VulkanPostProcessing.h"

#include <array>

#include "gr_vulkan.h"
#include "VulkanBarrier.h"
#include "VulkanRenderer.h"
#include "VulkanShader.h"
#include "VulkanDescriptorManager.h"
#include "graphics/2d.h"
#include "graphics/post_processing.h"
#include "osapi/outwnd.h"


namespace graphics::vulkan {

// ===== Lens Flare Pipeline Implementation =====

static constexpr uint32_t MAX_LENS_FLARES = 512;

// GPU-side flare entry (must match GLSL std430 layout exactly)
struct GpuFlareEntry {
	float meanPosX, meanPosY;   // offset  0, size 8
	float aabbMinX, aabbMinY;   // offset  8, size 8
	float aabbMaxX, aabbMaxY;   // offset 16, size 8
	float brightness;           // offset 24, size 4
	uint32_t mipLevel;          // offset 28, size 4
};
static_assert(sizeof(GpuFlareEntry) == 32, "GpuFlareEntry size mismatch");

struct GpuFlareSSBO {
	uint32_t flareCount;         // offset 0
	uint32_t pad;                // offset 4 (std430 struct alignment pad)
	GpuFlareEntry entries[MAX_LENS_FLARES]; // offset 8
};
static_assert(offsetof(GpuFlareSSBO, entries) == 8, "GpuFlareSSBO entries offset mismatch");
static_assert(sizeof(GpuFlareSSBO) == 8 + MAX_LENS_FLARES * sizeof(GpuFlareEntry), "GpuFlareSSBO size mismatch");

// Params UBO for compute shader (std140 layout)
struct alignas(16) FlareParamsUBO {
	int32_t  coarseMip;       // N
	int32_t  fineMip;         // M
	float    threshold;
	int32_t  maxFlares;
	float    invWidth;        // 1.0 / full-res width
	float    invHeight;       // 1.0 / full-res height
	float    _pad0;
	float    _pad1;
};
static_assert(sizeof(FlareParamsUBO) == 32, "FlareParamsUBO size mismatch");

bool VulkanLensFlare::init(PostProcessContext& ctx, const RenderTarget& sceneColor)
{
	m_ctx = &ctx;
	m_sceneColor = &sceneColor;

	m_mipLevelCount = static_cast<uint32_t>(m_coarseMip + 1);

	// Create lens-flare render pass (color-only RGBA16F, loadOp=eLoad, additive blend)
	{
		vk::AttachmentDescription att;
		att.format = HDR_COLOR_FORMAT;
		att.samples = vk::SampleCountFlagBits::e1;
		att.loadOp = vk::AttachmentLoadOp::eLoad;
		att.storeOp = vk::AttachmentStoreOp::eStore;
		att.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		att.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		att.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
		att.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

		vk::AttachmentReference colorRef;
		colorRef.attachment = 0;
		colorRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass;
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorRef;

		vk::SubpassDependency dep;
		dep.srcSubpass = VK_SUBPASS_EXTERNAL;
		dep.dstSubpass = 0;
		dep.srcStageMask = vk::PipelineStageFlagBits::eFragmentShader
		                  | vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dep.dstStageMask = vk::PipelineStageFlagBits::eFragmentShader
		                  | vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dep.srcAccessMask = vk::AccessFlagBits::eShaderRead
		                  | vk::AccessFlagBits::eColorAttachmentWrite;
		dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead
		                  | vk::AccessFlagBits::eColorAttachmentWrite;

		vk::RenderPassCreateInfo rpInfo;
		rpInfo.attachmentCount = 1;
		rpInfo.pAttachments = &att;
		rpInfo.subpassCount = 1;
		rpInfo.pSubpasses = &subpass;
		rpInfo.dependencyCount = 1;
		rpInfo.pDependencies = &dep;

		try {
			m_flareRenderPass = m_ctx->device.createRenderPass(rpInfo);
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to create render pass: %s\n", e.what()));
			return false;
		}
	}

	if (!createTargets()) {
		return false;
	}

	Gr_lensflare_coarse_mip = m_coarseMip;
	Gr_lensflare_fine_mip   = m_fineMip;

	if (!createComputeResources()) {
		destroyTargets();
		return false;
	}

	m_initialized = true;
	mprintf(("VulkanLensFlare: Initialized (%ux%u, coarseMip=%d, fineMip=%d, threshold=%.1f)\n",
		m_mipWidth, m_mipHeight, m_coarseMip, m_fineMip, m_threshold));
	return true;
}

void VulkanLensFlare::shutdown()
{
	if (!m_initialized) {
		return;
	}

	m_ctx->device.waitIdle();

	destroyComputeResources();
	destroyTargets();

	if (m_flareRenderPass) {
		m_ctx->device.destroyRenderPass(m_flareRenderPass);
		m_flareRenderPass = nullptr;
	}

	m_initialized = false;
	mprintf(("VulkanLensFlare: Shutdown complete\n"));
}

bool VulkanLensFlare::resize()
{
	destroyTargets();
	if (!createTargets()) {
		return false;
	}
	return true;
}

bool VulkanLensFlare::createTargets()
{
	m_mipWidth = m_ctx->sceneExtent.width;
	m_mipHeight = m_ctx->sceneExtent.height;

	vk::ImageCreateInfo imageInfo;
	imageInfo.imageType = vk::ImageType::e2D;
	imageInfo.format = HDR_COLOR_FORMAT;
	imageInfo.extent.width = m_mipWidth;
	imageInfo.extent.height = m_mipHeight;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = m_mipLevelCount;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = vk::SampleCountFlagBits::e1;
	imageInfo.tiling = vk::ImageTiling::eOptimal;
	imageInfo.usage = vk::ImageUsageFlagBits::eTransferSrc
	                | vk::ImageUsageFlagBits::eTransferDst
	                | vk::ImageUsageFlagBits::eSampled;
	imageInfo.sharingMode = vk::SharingMode::eExclusive;
	imageInfo.initialLayout = vk::ImageLayout::eUndefined;

	try {
		m_mipImage = m_ctx->device.createImage(imageInfo);
	} catch (const vk::SystemError& e) {
		mprintf(("VulkanLensFlare: Failed to create mip image: %s\n", e.what()));
		return false;
	}

	if (!m_ctx->memoryManager->allocateImageMemory(m_mipImage, MemoryUsage::GpuOnly, m_mipAllocation)) {
		mprintf(("VulkanLensFlare: Failed to allocate mip image memory!\n"));
		m_ctx->device.destroyImage(m_mipImage);
		m_mipImage = nullptr;
		return false;
	}

	// Full image view (all mip levels, for imageLoad in compute)
	vk::ImageViewCreateInfo fullViewInfo;
	fullViewInfo.image = m_mipImage;
	fullViewInfo.viewType = vk::ImageViewType::e2D;
	fullViewInfo.format = HDR_COLOR_FORMAT;
	fullViewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	fullViewInfo.subresourceRange.baseMipLevel = 0;
	fullViewInfo.subresourceRange.levelCount = m_mipLevelCount;
	fullViewInfo.subresourceRange.baseArrayLayer = 0;
	fullViewInfo.subresourceRange.layerCount = 1;

	try {
		m_mipFullView = m_ctx->device.createImageView(fullViewInfo);
	} catch (const vk::SystemError& e) {
		mprintf(("VulkanLensFlare: Failed to create mip image view: %s\n", e.what()));
		return false;
	}

	// Create flare framebuffer (wraps scene color for additive compositing)
	{
		vk::FramebufferCreateInfo fbInfo;
		fbInfo.renderPass = m_flareRenderPass;
		fbInfo.attachmentCount = 1;
		fbInfo.pAttachments = &m_sceneColor->view;
		fbInfo.width = m_ctx->sceneExtent.width;
		fbInfo.height = m_ctx->sceneExtent.height;
		fbInfo.layers = 1;

		try {
			m_flareFramebuffer = m_ctx->device.createFramebuffer(fbInfo);
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to create flare framebuffer: %s\n", e.what()));
			return false;
		}
	}

	return true;
}

void VulkanLensFlare::destroyTargets()
{
	if (m_flareFramebuffer) {
		m_ctx->device.destroyFramebuffer(m_flareFramebuffer);
		m_flareFramebuffer = nullptr;
	}
	if (m_mipFullView) {
		m_ctx->device.destroyImageView(m_mipFullView);
		m_mipFullView = nullptr;
	}
	if (m_mipImage) {
		m_ctx->device.destroyImage(m_mipImage);
		m_mipImage = nullptr;
	}
	if (m_mipAllocation.isValid()) {
		m_ctx->memoryManager->freeAllocation(m_mipAllocation);
		m_mipAllocation = {};
	}
}

bool VulkanLensFlare::createComputeResources()
{
	auto* shaderMgr = getShaderManager();

	// ---- Descriptor set layout (1 set, 3 bindings) ----
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings;

	bindings[0].binding = 0;
	bindings[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
	bindings[0].pImmutableSamplers = nullptr;

	bindings[1].binding = 1;
	bindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
	bindings[1].pImmutableSamplers = nullptr;

	bindings[2].binding = 2;
	bindings[2].descriptorType = vk::DescriptorType::eUniformBuffer;
	bindings[2].descriptorCount = 1;
	bindings[2].stageFlags = vk::ShaderStageFlagBits::eCompute;
	bindings[2].pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutCreateInfo dslInfo;
	dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	dslInfo.pBindings = bindings.data();
	try {
		m_computeDescSetLayout = m_ctx->device.createDescriptorSetLayout(dslInfo);
	} catch (const vk::SystemError& e) {
		mprintf(("VulkanLensFlare: Failed to create compute descriptor set layout: %s\n", e.what()));
		return false;
	}

	// ---- Pipeline layout ----
	vk::PipelineLayoutCreateInfo plInfo;
	plInfo.setLayoutCount = 1;
	plInfo.pSetLayouts = &m_computeDescSetLayout;
	plInfo.pushConstantRangeCount = 0;
	plInfo.pPushConstantRanges = nullptr;

	try {
		m_computePipelineLayout = m_ctx->device.createPipelineLayout(plInfo);
	} catch (const vk::SystemError& e) {
		mprintf(("VulkanLensFlare: Failed to create compute pipeline layout: %s\n", e.what()));
		return false;
	}

	// ---- Compute shader module ----
	auto computeModule = shaderMgr->createComputeModule("lensflare-detect-c.sdr",
	                                                    SDR_TYPE_POST_PROCESS_LENSFLARE_DETECT, 0);
	if (!computeModule) {
		mprintf(("VulkanLensFlare: Failed to create compute shader module\n"));
		return false;
	}
	m_computeModule = std::move(computeModule);

	// ---- Compute pipeline ----
	vk::PipelineShaderStageCreateInfo stageInfo;
	stageInfo.stage = vk::ShaderStageFlagBits::eCompute;
	stageInfo.module = m_computeModule.get();
	stageInfo.pName = "main";

	vk::ComputePipelineCreateInfo cpInfo;
	cpInfo.stage = stageInfo;
	cpInfo.layout = m_computePipelineLayout;
	cpInfo.basePipelineHandle = nullptr;
	cpInfo.basePipelineIndex = -1;

	try {
		auto result = m_ctx->device.createComputePipeline(nullptr, cpInfo);
		m_computePipeline = result.value;
	} catch (const vk::SystemError& e) {
		mprintf(("VulkanLensFlare: Failed to create compute pipeline: %s\n", e.what()));
		return false;
	}

	// ---- Flare SSBO (persistently mapped for CPU reset) ----
	{
		vk::DeviceSize bufSize = sizeof(GpuFlareSSBO);

		vk::BufferCreateInfo bufInfo;
		bufInfo.size = bufSize;
		bufInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer
		              | vk::BufferUsageFlagBits::eTransferDst
		              | vk::BufferUsageFlagBits::eTransferSrc;
		bufInfo.sharingMode = vk::SharingMode::eExclusive;

		try {
			m_flareBuffer = m_ctx->device.createBuffer(bufInfo);
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to create flare SSBO: %s\n", e.what()));
			return false;
		}

		if (!m_ctx->memoryManager->allocateBufferMemory(m_flareBuffer, MemoryUsage::CpuToGpu, m_flareBufferAlloc)) {
			mprintf(("VulkanLensFlare: Failed to allocate flare SSBO memory!\n"));
			return false;
		}

		m_flareBufferMapped = m_ctx->memoryManager->mapMemory(m_flareBufferAlloc);
		if (!m_flareBufferMapped) {
			mprintf(("VulkanLensFlare: Failed to map flare SSBO memory!\n"));
			return false;
		}
	}

	// ---- Params UBO (persistently mapped for CPU update) ----
	{
		vk::DeviceSize bufSize = sizeof(FlareParamsUBO);

		vk::BufferCreateInfo bufInfo;
		bufInfo.size = bufSize;
		bufInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
		bufInfo.sharingMode = vk::SharingMode::eExclusive;

		try {
			m_paramsBuffer = m_ctx->device.createBuffer(bufInfo);
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to create params UBO: %s\n", e.what()));
			return false;
		}

		if (!m_ctx->memoryManager->allocateBufferMemory(m_paramsBuffer, MemoryUsage::CpuToGpu, m_paramsAlloc)) {
			mprintf(("VulkanLensFlare: Failed to allocate params UBO memory!\n"));
			return false;
		}

		m_paramsMapped = m_ctx->memoryManager->mapMemory(m_paramsAlloc);
		if (!m_paramsMapped) {
			mprintf(("VulkanLensFlare: Failed to map params UBO memory!\n"));
			return false;
		}
	}

	// ---- Descriptor pool + set ----
	{
		std::array<vk::DescriptorPoolSize, 3> poolSizes;
		poolSizes[0].type = vk::DescriptorType::eCombinedImageSampler;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
		poolSizes[1].descriptorCount = 1;
		poolSizes[2].type = vk::DescriptorType::eUniformBuffer;
		poolSizes[2].descriptorCount = 1;

		vk::DescriptorPoolCreateInfo poolInfo;
		poolInfo.maxSets = 1;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();

		try {
			m_computeDescPool = m_ctx->device.createDescriptorPool(poolInfo);
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to create descriptor pool: %s\n", e.what()));
			return false;
		}

		vk::DescriptorSetAllocateInfo allocInfo;
		allocInfo.descriptorPool = m_computeDescPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &m_computeDescSetLayout;

		try {
			m_computeDescSet = m_ctx->device.allocateDescriptorSets(allocInfo).front();
		} catch (const vk::SystemError& e) {
			mprintf(("VulkanLensFlare: Failed to allocate descriptor set: %s\n", e.what()));
			return false;
		}

		// Write descriptor set
		vk::DescriptorImageInfo imageInfo;
		imageInfo.imageView = m_mipFullView;
		imageInfo.sampler = m_ctx->mipmapSampler;
		imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo.buffer = m_flareBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = VK_WHOLE_SIZE;

		vk::DescriptorBufferInfo paramInfo;
		paramInfo.buffer = m_paramsBuffer;
		paramInfo.offset = 0;
		paramInfo.range = VK_WHOLE_SIZE;

		std::array<vk::WriteDescriptorSet, 3> writes;
		writes[0].dstSet = m_computeDescSet;
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
		writes[0].pImageInfo = &imageInfo;

		writes[1].dstSet = m_computeDescSet;
		writes[1].dstBinding = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
		writes[1].pBufferInfo = &bufferInfo;

		writes[2].dstSet = m_computeDescSet;
		writes[2].dstBinding = 2;
		writes[2].descriptorCount = 1;
		writes[2].descriptorType = vk::DescriptorType::eUniformBuffer;
		writes[2].pBufferInfo = &paramInfo;

		m_ctx->device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	return true;
}

void VulkanLensFlare::destroyComputeResources()
{
	if (m_flareBufferMapped && m_flareBufferAlloc.isValid()) {
		m_ctx->memoryManager->unmapMemory(m_flareBufferAlloc);
		m_flareBufferMapped = nullptr;
	}
	if (m_paramsMapped && m_paramsAlloc.isValid()) {
		m_ctx->memoryManager->unmapMemory(m_paramsAlloc);
		m_paramsMapped = nullptr;
	}
	if (m_flareBuffer) {
		m_ctx->device.destroyBuffer(m_flareBuffer);
		m_flareBuffer = nullptr;
	}
	if (m_flareBufferAlloc.isValid()) {
		m_ctx->memoryManager->freeAllocation(m_flareBufferAlloc);
		m_flareBufferAlloc = {};
	}
	if (m_paramsBuffer) {
		m_ctx->device.destroyBuffer(m_paramsBuffer);
		m_paramsBuffer = nullptr;
	}
	if (m_paramsAlloc.isValid()) {
		m_ctx->memoryManager->freeAllocation(m_paramsAlloc);
		m_paramsAlloc = {};
	}
	if (m_computePipeline) {
		m_ctx->device.destroyPipeline(m_computePipeline);
		m_computePipeline = nullptr;
	}
	if (m_computeModule) {
		m_computeModule.reset();
	}
	if (m_computePipelineLayout) {
		m_ctx->device.destroyPipelineLayout(m_computePipelineLayout);
		m_computePipelineLayout = nullptr;
	}
	if (m_computeDescPool) {
		m_ctx->device.destroyDescriptorPool(m_computeDescPool);
		m_computeDescPool = nullptr;
	}
	if (m_computeDescSetLayout) {
		m_ctx->device.destroyDescriptorSetLayout(m_computeDescSetLayout);
		m_computeDescSetLayout = nullptr;
	}
}

void VulkanLensFlare::capturePreBloom(vk::CommandBuffer cmd)
{
	if (!m_initialized) {
		return;
	}

	GR_DEBUG_SCOPE("LensFlare capture pre-bloom");

	// 1. Copy scene color (pre-bloom) → mip chain level 0
	// Only transition level 0 so that other mips stay eUndefined for
	// generateMipmaps (which expects its destination levels to be
	// eUndefined before the first blit).
	copyImageToImage(cmd,
		m_sceneColor->image,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		m_mipImage,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		m_ctx->sceneExtent,
		vk::ImageAspectFlagBits::eColor,
		1);

	// 2. Generate mip chain (transitions all mips → eShaderReadOnlyOptimal)
	PostProcessContext::generateMipmaps(cmd, m_mipImage, m_mipWidth, m_mipHeight, m_mipLevelCount);

	// 3. Make mip chain visible to compute shader (same layout, stage dependency)
	{
		ImageBarrier2 barrier;
		barrier.image = m_mipImage;
		barrier.levelCount = m_mipLevelCount;
		barrier.layerCount = 1;
		barrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcStage = vk::PipelineStageFlagBits2::eTransfer;
		barrier.srcAccess = vk::AccessFlagBits2::eTransferWrite;
		barrier.dstStage = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.dstAccess = vk::AccessFlagBits2::eShaderSampledRead;
		cmdImageBarrier(cmd, barrier);
	}
}

void VulkanLensFlare::execute(vk::CommandBuffer cmd)
{
	if (!m_initialized) {
		return;
	}

	GR_DEBUG_SCOPE("LensFlare detect & render");

	{
		static int logFrame = 0;
		if (++logFrame % 60 == 1) {
			auto* ssbo = static_cast<GpuFlareSSBO*>(m_flareBufferMapped);
			mprintf(("LensFlare: prevDetect=%u coarseMip=%d fineMip=%d threshold=%.2f\n",
				ssbo->flareCount, m_coarseMip, m_fineMip, m_threshold));
			if (ssbo->flareCount > 0) {
				uint32_t n = std::min(ssbo->flareCount, 3u);
				for (uint32_t i = 0; i < n; ++i) {
					auto& e = ssbo->entries[i];
					mprintf(("  [%u] mean=(%.3f,%.3f) aabb=(%.3f,%.3f)-(%.3f,%.3f) lum=%.1f mip=%u\n",
						i, e.meanPosX, e.meanPosY,
						e.aabbMinX, e.aabbMinY, e.aabbMaxX, e.aabbMaxY,
						e.brightness, e.mipLevel));
				}
			}
		}
	}

	// ================================================================
	// 1. Zero the flare count + entries via GPU fill
	// ================================================================
	{
		vk::DeviceSize ssboSize = sizeof(GpuFlareEntry) * MAX_LENS_FLARES + 8;
		cmd.fillBuffer(m_flareBuffer, 0, ssboSize, 0);
	}

	// ================================================================
	// 2. Make the zeroed SSBO visible to the compute shader
	// ================================================================
	{
		vk::MemoryBarrier2 barrier;
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
		barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite
		                      | vk::AccessFlagBits2::eShaderStorageRead;

		vk::DependencyInfo dep;
		dep.memoryBarrierCount = 1;
		dep.pMemoryBarriers = &barrier;
		cmd.pipelineBarrier2(dep);
	}

	// ================================================================
	// 3. Update params UBO (mapped write)
	// ================================================================
	{
		FlareParamsUBO params;
		params.coarseMip  = static_cast<int32_t>(m_coarseMip);
		params.fineMip    = static_cast<int32_t>(m_fineMip);
		params.threshold  = m_threshold;
		params.maxFlares  = static_cast<int32_t>(MAX_LENS_FLARES);
		params.invWidth   = 1.0f / static_cast<float>(m_mipWidth);
		params.invHeight  = 1.0f / static_cast<float>(m_mipHeight);
		params._pad0      = 0.0f;
		params._pad1      = 0.0f;

		memcpy(m_paramsMapped, &params, sizeof(params));
	}

	// ================================================================
	// 4. Dispatch compute shader
	// ================================================================
	{
		GR_DEBUG_SCOPE("LensFlare detect");

		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_computePipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_computePipelineLayout,
		                       0, 1, &m_computeDescSet, 0, nullptr);

		uint32_t div      = 1u << m_coarseMip;
		uint32_t csWidth  = std::max(1u, m_mipWidth  / div);
		uint32_t csHeight = std::max(1u, m_mipHeight / div);
		uint32_t gx       = (csWidth  + 7) / 8;
		uint32_t gy       = (csHeight + 7) / 8;

		cmd.dispatch(gx, gy, 1);
	}

	// ================================================================
	// 5. Barrier: compute shader writes → vertex shader reads (SSBO)
	// ================================================================
	{
		vk::MemoryBarrier2 barrier;
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eVertexShader;
		barrier.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead;

		vk::DependencyInfo dep;
		dep.memoryBarrierCount = 1;
		dep.pMemoryBarriers = &barrier;
		cmd.pipelineBarrier2(dep);
	}

	// ================================================================
	// 6. Transition scene color → eColorAttachmentOptimal for compositing
	// ================================================================
	{
		ImageBarrier2 barrier;
		barrier.image = m_sceneColor->image;
		barrier.levelCount = 1;
		barrier.layerCount = 1;
		barrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
		barrier.srcStage = vk::PipelineStageFlagBits2::eFragmentShader;
		barrier.srcAccess = vk::AccessFlagBits2::eShaderSampledRead;
		barrier.dstStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
		barrier.dstAccess = vk::AccessFlagBits2::eColorAttachmentRead
		                  | vk::AccessFlagBits2::eColorAttachmentWrite;
		cmdImageBarrier(cmd, barrier);
	}

	// ================================================================
	// 7. Instanced flare quad rendering
	// ================================================================
	{
		GR_DEBUG_SCOPE("LensFlare render");

		auto* pipelineMgr = getPipelineManager();
		auto* descMgr     = getDescriptorManager();

		if (!pipelineMgr || !descMgr) {
			goto restoreSceneLayout;
		}

		PipelineConfig config;
		config.shaderType = SDR_TYPE_POST_PROCESS_LENSFLARE_RENDER;
		config.shaderFlags = 0;
		config.vertexLayoutHash = 0;
		config.primitiveType = PRIM_TYPE_TRIS;
		config.depthMode = ZBUFFER_TYPE_NONE;
		config.blendMode = ALPHA_BLEND_ADDITIVE;
		config.cullEnabled = false;
		config.depthWriteEnabled = false;
		config.renderPass = m_flareRenderPass;
		config.colorAttachmentCount = 1;

		vertex_layout emptyLayout;
		vk::Pipeline pipeline = pipelineMgr->getPipeline(config, emptyLayout);
		if (!pipeline) {
			mprintf(("VulkanLensFlare: Failed to get render pipeline!\n"));
			goto restoreSceneLayout;
		}

		vk::PipelineLayout pipelineLayout = pipelineMgr->getPipelineLayout();

		vk::RenderPassBeginInfo rpBegin;
		rpBegin.renderPass = m_flareRenderPass;
		rpBegin.framebuffer = m_flareFramebuffer;
		rpBegin.renderArea.offset = vk::Offset2D(0, 0);
		rpBegin.renderArea.extent = m_ctx->sceneExtent;

		cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		{
			vk::Viewport vp;
			vp.x = 0.0f;
			vp.y = 0.0f;
			vp.width = static_cast<float>(m_ctx->sceneExtent.width);
			vp.height = static_cast<float>(m_ctx->sceneExtent.height);
			vp.minDepth = 0.0f;
			vp.maxDepth = 1.0f;
			cmd.setViewport(0, vp);

			vk::Rect2D sc;
			sc.offset = vk::Offset2D(0, 0);
			sc.extent = m_ctx->sceneExtent;
			cmd.setScissor(0, sc);
		}

		{
			DescriptorWriter writer;
			writer.reset(m_ctx->device, descMgr->getFallbacks());

			vk::DescriptorSet materialSet = descMgr->allocateFrameSet(DescriptorSetIndex::Material);
			Assert(materialSet);
			writer.writeSet(materialSet,
			                VulkanDescriptorManager::getSetTemplate(DescriptorSetIndex::Material));
			{
				vk::DescriptorBufferInfo flareInfo;
				flareInfo.buffer = m_flareBuffer;
				flareInfo.offset = 0;
				flareInfo.range = VK_WHOLE_SIZE;
				writer.setBuffer(MaterialBinding::FlareData, flareInfo);
			}

			vk::DescriptorSet perDrawSet = descMgr->allocateFrameSet(DescriptorSetIndex::PerDraw);
			Assert(perDrawSet);
			writer.writeSet(perDrawSet,
			                VulkanDescriptorManager::getSetTemplate(DescriptorSetIndex::PerDraw));

			writer.flush();

			// Global set (Set 0) is already bound from frame setup.
			// Only bind Material + PerDraw sets.
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
			                       static_cast<uint32_t>(DescriptorSetIndex::Material),
			                       {materialSet, perDrawSet}, {});
		}

		cmd.draw(6, static_cast<uint32_t>(MAX_LENS_FLARES), 0, 0);

		cmd.endRenderPass();
	}

	// ================================================================
	// 8. Transition scene color back → eShaderReadOnlyOptimal
	// ================================================================
	{
		ImageBarrier2 barrier;
		barrier.image = m_sceneColor->image;
		barrier.levelCount = 1;
		barrier.layerCount = 1;
		barrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
		barrier.srcAccess = vk::AccessFlagBits2::eColorAttachmentWrite;
		barrier.dstStage = vk::PipelineStageFlagBits2::eFragmentShader;
		barrier.dstAccess = vk::AccessFlagBits2::eShaderSampledRead;
		cmdImageBarrier(cmd, barrier);
	}

	return;

restoreSceneLayout:
	{
		ImageBarrier2 barrier;
		barrier.image = m_sceneColor->image;
		barrier.levelCount = 1;
		barrier.layerCount = 1;
		barrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
		barrier.srcAccess = vk::AccessFlagBits2::eColorAttachmentWrite;
		barrier.dstStage = vk::PipelineStageFlagBits2::eFragmentShader;
		barrier.dstAccess = vk::AccessFlagBits2::eShaderSampledRead;
		cmdImageBarrier(cmd, barrier);
	}
}

} // namespace graphics::vulkan
