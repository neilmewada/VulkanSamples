#pragma once
#include <stdint.h>

#ifdef _WIN32
  #include <windows.h>
#endif

// Bump if you add/remove fields
#define CUDA_IMPORT_PACK_VERSION 1

// Use your own enum if you don't want to include <vulkan/vulkan.h> on the CUDA side.
// Store VkFormat as a raw u32 to avoid header coupling.
struct CudaImportPack {
  uint32_t version;           // = CUDA_IMPORT_PACK_VERSION
  uint32_t width;             // image width in pixels
  uint32_t height;            // image height in pixels
  uint32_t vk_format;         // numeric VkFormat (e.g., VK_FORMAT_R8G8B8A8_UNORM)

  // Vulkan allocation info (for cudaImportExternalMemory)
  uint64_t alloc_size;        // EXACT VkMemoryRequirements.size
  uint64_t alloc_offset;      // usually 0 (dedicated alloc); keep for completeness
  uint8_t  dedicated;         // 1 if VkMemoryDedicatedAllocateInfo was used
  uint8_t  reserved_[7];      // alignment/padding to 8-byte boundary

#ifdef _WIN32
  HANDLE   memory_handle;     // from vkGetMemoryWin32HandleKHR (OPAQUE_WIN32)
  HANDLE   semaphore_handle;  // from vkGetSemaphoreWin32HandleKHR (timeline or binary)
#else
  int      memory_fd;         // from vkGetMemoryFdKHR (OPAQUE_FD)
  int      semaphore_fd;      // from vkGetSemaphoreFdKHR
#endif

  // Semaphore mode
  uint8_t  semaphore_is_timeline; // 1 = timeline, 0 = binary
  uint8_t  reserved2_[7];         // padding
};
