#define GLFW_INCLUDE_VULKAN
#include <array>
#include <GLFW/glfw3.h>
#include "vkapp.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define ERROR_LOG(err) printf("ERROR: %s\n", err);

#define MAX_FRAMES_IN_FLIGHT 2

struct UniformBufferObject{
    std::array<float, 16> mvp;
};

std::vector<uint8_t> load_shader(const char *file_path) {
    std::vector<uint8_t> file_content;
#if _MSC_VER
	FILE* fptr;
	fopen_s(&fptr, file_path, "rb");
#else
	FILE* fptr = fopen(file_path, "rb");
#endif
	if(!fptr)
	{
		ERROR_LOG("failed to open spirv file");
		return file_content;
	}

	fseek(fptr, 0, SEEK_END);
	size_t file_size = ftell(fptr);

	fseek(fptr, 0, SEEK_SET);

	file_content.resize(file_size);
	fread(file_content.data(), file_size, 1, fptr);

	fclose(fptr);
    return file_content;
}

struct VulkanApp {
    GLFWwindow* window;
    VkInstance instance;
    VkSurfaceKHR surface;

    VkDevice device;
	VkPhysicalDevice physicalDevice;
    uint32_t graphicsComputeFamilyIdx_;
    int presentFamilyIdx_;
    VkQueue graphicsQueue;
	VkQueue presentQueue;

    VkSwapchainKHR swapchain;
	VkFormat swapchainFormat;
	VkExtent2D swapchainExtent;
	uint32_t swapchainImageCount;
	VkImage* swapchainImages;
	VkImageView* swapchainImageViews;

    VkRenderPass renderPass;

    uint32_t framebufferCount;
	VkFramebuffer* framebuffers;

    VkDescriptorSetLayout descriptorLayout;
	VkPipelineLayout layout;
	VkPipeline pipeline;

    VkDescriptorPool pool;
    VkDescriptorSet* sets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkCommandPool commandPool;
	VkCommandBuffer commandBuffers[MAX_FRAMES_IN_FLIGHT];

	VkSemaphore imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT];
	VkSemaphore renderFinishedSemaphores[MAX_FRAMES_IN_FLIGHT];
	VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];

    VkSurfaceTransformFlagBitsKHR pretransformFlag;
    float grey = 0.0f;

    int w = 1280, h = 720;

    void run() {
        initWindow();
        initVulkan();
        runMainLoop();
    }

    void runMainLoop() {
        float lastTime = (float)glfwGetTime();

        float accumTime = 0.0f;
        uint32_t accumFrames = 0;

        while(!glfwWindowShouldClose(window))
        {
            float curTime = (float)glfwGetTime();
            float dt = curTime - lastTime;
            lastTime = curTime;

            accumTime += dt;
            accumFrames++;

            if(accumTime >= 1.0f)
            {
                float avgDt = accumTime / accumFrames;

                char windowName[64];
                snprintf(windowName, sizeof(windowName), "super vulkan example [FPS: %.0f (%.2fms)]", 1.0f / avgDt, avgDt * 1000.0f);
                glfwSetWindowTitle(window, windowName);

                accumTime -= 1.0f;
                accumFrames = 0;
            }

            render(dt);
        
            glfwPollEvents();
        }
    }

    void initWindow() {
        const char* name = "super vulkan example";

        if(!glfwInit())
        {
            printf("failed to initialize GLFW");
            return;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(w, h, name, NULL, NULL);
        if(!window)
        {
            printf("failed to create GLFW window");
            return;
        }
    }

    void initVulkan() {
        initInstance();
        initSurface();
        initPhysicalDevice();
        initLogicalDevice();
        initSwapChain();
        initRenderPass();
        initFramebuffers();
        initDescriptorLayout();
        initUniformBuffers();
        initDescriptorPool();
        initDescriptorSets();
        initGraphicsPipeline();
        initCommandPool();
        initCommandBuffer();
        initSyncObjects();
    }

    void initInstance() {
        uint32_t requiredExtensionCount;
        char** requiredGlfwExtensions = (char**)glfwGetRequiredInstanceExtensions(&requiredExtensionCount);
        if(!requiredGlfwExtensions)
        {
            ERROR_LOG("Vulkan rendering not supported on this machine");
            return;
        }

        VkApplicationInfo appInfo = {0}; //most of this stuff is pretty useless, just for drivers to optimize certain programs
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Triangle App";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);

        VkInstanceCreateInfo instanceInfo = {0};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        instanceInfo.enabledExtensionCount =
            static_cast<uint32_t>(requiredExtensionCount);
        instanceInfo.ppEnabledExtensionNames = requiredGlfwExtensions;

        if(vkCreateInstance(&instanceInfo, NULL, &instance) != VK_SUCCESS)
        {
            ERROR_LOG("failed to create Vulkan instance");
            return;
        }
    }

    void initSurface() {
        if(glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS)
        {
            ERROR_LOG("failed to create window surface");
            return;
        }
    }

    void initPhysicalDevice() {
        uint32_t deviceCount;
	    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

        if(deviceCount == 0)
        {
            ERROR_LOG("failed to find a physical device that supports Vulkan");
            return;
        }

        VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

        for(uint32_t i = 0; i < deviceCount; i++)
        {

            VkPhysicalDeviceProperties properties;
            VkPhysicalDeviceFeatures features;
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceProperties      (devices[i], &properties);
            vkGetPhysicalDeviceFeatures        (devices[i], &features);
            vkGetPhysicalDeviceMemoryProperties(devices[i], &memProperties);

            printf("Device %d: %s\n", i, properties.deviceName);

            //check if required queue families are supported:
            //---------------
            int32_t graphicsComputeFamilyIdx = -1;
            int32_t presentFamilyIdx = -1;

            uint32_t queueFamilyCount;
            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, NULL);
            VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, queueFamilies);

            for(uint32_t j = 0; j < queueFamilyCount; j++)
            {
                if((queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (queueFamilies[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) //TODO: see if there is a way to determine most optimal queue families
                    graphicsComputeFamilyIdx = j;
                
                VkBool32 presentSupport;
                vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], j, surface, &presentSupport);
                if(presentSupport)
                    presentFamilyIdx = j;

                if(graphicsComputeFamilyIdx > 0 && presentFamilyIdx > 0)
                    break;
            }

            free(queueFamilies);

            if(graphicsComputeFamilyIdx < 0 || presentFamilyIdx < 0)
                continue;

            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(devices[i], NULL, &extensionCount, NULL);
            VkExtensionProperties* extensions = (VkExtensionProperties*)malloc(extensionCount * sizeof(VkExtensionProperties));
            vkEnumerateDeviceExtensionProperties(devices[i], NULL, &extensionCount, extensions);

            free(extensions);
            if(i == 0) {
                physicalDevice = devices[0];
                graphicsComputeFamilyIdx_ = graphicsComputeFamilyIdx;
			    presentFamilyIdx_ = presentFamilyIdx;
            }
        }

	    free(devices);
    }

    void initLogicalDevice() {
        std::vector<const char*> device_extensions;
        device_extensions.push_back("VK_KHR_swapchain");

        uint32_t queueCount = 0;
        uint32_t queueIndices[2];
        if(graphicsComputeFamilyIdx_ == presentFamilyIdx_)
        {
            queueCount = 1;
            queueIndices[0] = graphicsComputeFamilyIdx_;
        }
        else
        {
            queueCount = 2;
            queueIndices[0] = graphicsComputeFamilyIdx_;
            queueIndices[1] = presentFamilyIdx_;
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfos[2];
        for(uint32_t i = 0; i < queueCount; i++)
        {
            VkDeviceQueueCreateInfo queueInfo = {0};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queueIndices[i]; //TODO: see how we can optimize this, when would we want multiple queues?
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;

            queueInfos[i] = queueInfo;
        }

        //set features:
        //---------------
        VkPhysicalDeviceFeatures features = {0}; //TODO: allow wanted features to be passed in
        features.samplerAnisotropy = VK_TRUE;

        //create device:
        //---------------
        VkDeviceCreateInfo deviceInfo = {0};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = queueCount;
        deviceInfo.pQueueCreateInfos = queueInfos;
        deviceInfo.pEnabledFeatures = &features;
        deviceInfo.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        deviceInfo.ppEnabledExtensionNames = device_extensions.data();

        if(vkCreateDevice(physicalDevice, &deviceInfo, NULL, &device) != VK_SUCCESS)
        {
            ERROR_LOG("failed to create Vulkan device");
            return;
        }

        vkGetDeviceQueue(device, graphicsComputeFamilyIdx_, 0, &graphicsQueue);
        vkGetDeviceQueue(device, presentFamilyIdx_, 0,         &presentQueue);
    }

    void initSwapChain() {
        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                                &surfaceCapabilities);
        // Query the list of supported surface format and choose one we like
        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface,
                                            &formatCount, nullptr);
        VkSurfaceFormatKHR* formats = new VkSurfaceFormatKHR[formatCount];
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface,
                                            &formatCount, formats);

        uint32_t chosenFormat;
        for (chosenFormat = 0; chosenFormat < formatCount; chosenFormat++) {
            if (formats[chosenFormat].format == VK_FORMAT_B8G8R8A8_SRGB &&
                formats[chosenFormat].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) break;
        }

        swapchainFormat = formats[chosenFormat].format;

        VkSurfaceCapabilitiesKHR surfaceCap;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice,
                                                        surface, &surfaceCap);
        // assert(surfaceCap.supportedCompositeAlpha | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);

        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 &&
            imageCount > surfaceCapabilities.maxImageCount) {
            imageCount = surfaceCapabilities.maxImageCount;
        }
        

        VkExtent2D extent;
        if(surfaceCapabilities.currentExtent.width != UINT32_MAX)
        {
            //TODO: does this work when the window is resized?
            extent = surfaceCapabilities.currentExtent; //window already defined size for us
        }
        else
        {
            extent.width = w;
            extent.height = h;
            
            //clamping:
            extent.width = extent.width > surfaceCapabilities.maxImageExtent.width ? surfaceCapabilities.maxImageExtent.width : extent.width;
            extent.width = extent.width < surfaceCapabilities.minImageExtent.width ? surfaceCapabilities.minImageExtent.width : extent.width;

            extent.height = extent.height > surfaceCapabilities.maxImageExtent.height ? surfaceCapabilities.maxImageExtent.height : extent.height;
            extent.height = extent.height < surfaceCapabilities.minImageExtent.height ? surfaceCapabilities.minImageExtent.height : extent.height;
        }

        VkSwapchainCreateInfoKHR swapchainCreateInfo;
        swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainCreateInfo.pNext = nullptr;
        swapchainCreateInfo.surface = surface;
        swapchainCreateInfo.minImageCount = imageCount;
        swapchainCreateInfo.imageFormat = formats[chosenFormat].format;
        swapchainCreateInfo.imageColorSpace = formats[chosenFormat].colorSpace;
        swapchainCreateInfo.imageExtent = extent;
        swapchainCreateInfo.imageArrayLayers = 1;
        swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        swapchainCreateInfo.clipped = VK_FALSE;
        swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

        uint32_t indices[] = {graphicsComputeFamilyIdx_, presentFamilyIdx_};

        if(graphicsComputeFamilyIdx_ != presentFamilyIdx_)
        {
            swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapchainCreateInfo.queueFamilyIndexCount = 2;
            swapchainCreateInfo.pQueueFamilyIndices = indices;
        }
        else {
            swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr,
                                    &swapchain);

        swapchainExtent = extent;

        // Get the length of the created swap chain
        vkGetSwapchainImagesKHR(device, swapchain,
                                        &imageCount, nullptr);

        swapchainImageCount = imageCount;

        swapchainImages     =     (VkImage*)malloc(swapchainImageCount * sizeof(VkImage));
        swapchainImageViews = (VkImageView*)malloc(swapchainImageCount * sizeof(VkImageView));
        vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages);

        for(uint32_t i = 0; i < swapchainImageCount; i++) {
            VkImageViewCreateInfo viewInfo = {0};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = swapchainFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            VkImageView view;
            if(vkCreateImageView(device, &viewInfo, NULL, &view) != VK_SUCCESS) {
                ERROR_LOG("failed to create image view");
            }
            swapchainImageViews[i] = view;
        }
    }

    void initRenderPass() {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapchainFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentReference = {};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentReference;
        subpass.pDepthStencilAttachment = nullptr;

        VkSubpassDependency attachmentDependency = {};
        attachmentDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        attachmentDependency.dstSubpass = 0;
        attachmentDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        attachmentDependency.srcAccessMask = 0;
        attachmentDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        attachmentDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkSubpassDependency dependencies[1] = {attachmentDependency};

        const uint32_t attachmentCount = 1;
        VkAttachmentDescription attachments[1] = {colorAttachment};

        VkRenderPassCreateInfo renderPassCreateInfo = {};
        renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCreateInfo.attachmentCount = attachmentCount;
        renderPassCreateInfo.pAttachments = attachments;
        renderPassCreateInfo.subpassCount = 1;
        renderPassCreateInfo.pSubpasses = &subpass;
        renderPassCreateInfo.dependencyCount = 1;
        renderPassCreateInfo.pDependencies = dependencies;

        if(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            printf("failed to create final render pass\n");
            return;
        }
    }

    void initFramebuffers() {
        framebufferCount = swapchainImageCount;
        framebuffers = (VkFramebuffer*)malloc(framebufferCount * sizeof(VkFramebuffer));

        for(uint32_t i = 0; i < framebufferCount; i++)
        {
            VkImageView attachments[1] = {swapchainImageViews[i]};

            VkFramebufferCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            createInfo.renderPass = renderPass;
            createInfo.attachmentCount = 1;
            createInfo.pAttachments = attachments;
            createInfo.width = swapchainExtent.width;
            createInfo.height = swapchainExtent.height;
            createInfo.layers = 1;

            if(vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            {
                ERROR_LOG("failed to create framebuffer");
                return;
            }
        }
    }

    void initDescriptorLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                            &descriptorLayout);
    }

    void initDescriptorPool() {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
    }

    void initDescriptorSets() {
        VkDescriptorSetLayout* layouts = (VkDescriptorSetLayout*)malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkDescriptorSetLayout));
	    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		    layouts[i] = descriptorLayout;

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts;

        sets = (VkDescriptorSet*)malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkDescriptorSet));

        vkAllocateDescriptorSets(device, &allocInfo, sets);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = sets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter,
                                    VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                            properties) == properties) {
                return i;
            }
        }
        return -1;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags properties, VkBuffer &buffer,
                        VkDeviceMemory &bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties);

        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void initUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    VkShaderModule createShaderModule(const std::vector<uint8_t> &code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();

        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
        VkShaderModule shaderModule;
        vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);

        return shaderModule;
    }

    void initGraphicsPipeline() {
        auto vertShaderCode = load_shader("shader.vert.spv");
        auto fragShaderCode = load_shader("shader.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                            fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;

        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                        &layout);
        std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT,
                                                            VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicStateCI{};
        dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
        dynamicStateCI.dynamicStateCount =
            static_cast<uint32_t>(dynamicStateEnables.size());

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicStateCI;
        pipelineInfo.layout = layout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                            nullptr, &pipeline);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void initCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsComputeFamilyIdx_;
        vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    }

    void initCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        vkAllocateCommandBuffers(device, &allocInfo, commandBuffers);
    }

    void initSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                    &imageAvailableSemaphores[i]);

            vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                    &renderFinishedSemaphores[i]);

            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);
        }
    }

    void getPrerotationMatrix(const VkSurfaceTransformFlagBitsKHR &pretransformFlag,
                          std::array<float, 16> &mat) {
        // mat is initialized to the identity matrix
        mat = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
        if (pretransformFlag & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) {
            // mat is set to a 90 deg rotation matrix
            mat = {0., 1., 0., 0., -1., 0, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
        }

        else if (pretransformFlag & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
            // mat is set to 270 deg rotation matrix
            mat = {0., -1., 0., 0., 1., 0, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        UniformBufferObject ubo{};
        getPrerotationMatrix(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                            ubo.mvp);
        void *data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0,
                    &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void render(float deltaTime) {
        static uint32_t frameIdx = 0;

        vkWaitForFences(device, 1, &inFlightFences[frameIdx], VK_TRUE, UINT64_MAX);

        uint32_t imageIdx;
        VkResult imageAquireResult = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                        imageAvailableSemaphores[frameIdx], VK_NULL_HANDLE, &imageIdx);

        if(imageAquireResult == VK_ERROR_OUT_OF_DATE_KHR || imageAquireResult == VK_SUBOPTIMAL_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if(imageAquireResult != VK_SUCCESS)
        {
            ERROR_LOG("failed to acquire swapchain image");
            return;
        }

        updateUniformBuffer(frameIdx);
        vkResetFences(device, 1, &inFlightFences[frameIdx]);

        // vkh_copy_with_staging_buf(s->instance, s->cameraStagingBuffer, s->cameraStagingBufferMemory,
        //                                 s->cameraBuffers[frameIdx], sizeof(CameraGPU), 0, &camBuffer);

        vkResetCommandBuffer(commandBuffers[frameIdx], 0);

        recordCommandBuffer(commandBuffers[frameIdx], imageIdx, frameIdx, grey);

        grey += deltaTime/5.0f;

        if (grey > 1.0f) {
            grey = 0.0f;
        }

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphores[frameIdx];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[frameIdx];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores[frameIdx];

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameIdx]);

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[frameIdx];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIdx;
        presentInfo.pResults = nullptr;

        VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
        if(presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
            recreateSwapChain();
        else if(presentResult != VK_SUCCESS)
            ERROR_LOG("failed to present swapchain image");

        frameIdx = (frameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recreateSwapChain() {
        vkDeviceWaitIdle(device);
        cleanupSwapChain();
        initSwapChain();
        initFramebuffers();
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer,
                                  uint32_t imageIndex, uint32_t frameIdx, float grey) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchainExtent;

        VkViewport viewport{};
        viewport.width = (float)swapchainExtent.width;
        viewport.height = (float)swapchainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = swapchainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkClearValue clearColor = {{{grey, grey, grey, 1.0f}}};

        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
         vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                            VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            layout, 0, 1, &sets[frameIdx],
                            0, nullptr);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        vkEndCommandBuffer(commandBuffer);
    }

    void cleanupSwapChain() {
        for (size_t i = 0; i < framebufferCount; i++) {
            vkDestroyFramebuffer(device, framebuffers[i], nullptr);
        }

        for (size_t i = 0; i < swapchainImageCount; i++) {
            vkDestroyImageView(device, swapchainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapchain, nullptr);
    }
};


void run_vulkan_app() {
    VulkanApp app;
    app.run();
}