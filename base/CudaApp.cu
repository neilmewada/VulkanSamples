#include <cassert>
#include <cuda_runtime.h>

#include "interop_cuda_pack.h"

#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
	do {                                                                    \
		cudaError_t err = call;                                           \
		if (err != cudaSuccess) {                                         \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
					  << " code=" << static_cast<int>(err)                \
					  << " \"" << cudaGetErrorString(err) << "\""         \
					  << std::endl;                                       \
			std::exit(EXIT_FAILURE);                                      \
		}                                                                   \
	} while (0)

struct CudaImportedSlot {
	// Imported view of the Vulkan image memory
	cudaExternalMemory_t   extMem = nullptr;
	cudaMipmappedArray_t   mmArr = nullptr;
	cudaArray_t            level0 = nullptr;

	// Optional: a texture object to read texels from the image easily
	cudaTextureObject_t    tex = 0;

	// Imported semaphore that Vulkan signals, CUDA waits
	cudaExternalSemaphore_t extSem = nullptr;

	// Image info (handy for kernels later)
	int  width = 0;
	int  height = 0;
};

struct Nv12Surf
{
	uint8_t* base;
	size_t pitch;
	int w, h;
};

extern __global__ void bgra_to_nv12(cudaTextureObject_t tex,
	uint8_t* nv12Base, size_t pitch,
	int W, int H);

// Keep all slots here after import (size == number of exportable Vulkan images)
static std::vector<CudaImportedSlot> g_slots;
static std::vector<Nv12Surf> nv12;
static cudaStream_t g_stream = nullptr;

// ---------- 2) Import from a CudaImportPack (once per slot) ----------

extern "C" void cuda_interop_init()
{
	CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
}

extern "C" void cuda_interop_shutdown()
{
	if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

extern "C" void cuda_interop_sync()
{
	CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

static void import_from_pack(const CudaImportPack & p, CudaImportedSlot & s)
{
	s.width = static_cast<int>(p.width);
	s.height = static_cast<int>(p.height);

	// 2.a) Import Vulkan device memory as CUDA external memory
	cudaExternalMemoryHandleDesc md{};
#if defined(_WIN32)
	md.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	md.handle.win32.handle = p.memory_handle;       // HANDLE from vkGetMemoryWin32HandleKHR
#else
	md.type = cudaExternalMemoryHandleTypeOpaqueFd;
	md.handle.fd = p.memory_fd;           // FD from vkGetMemoryFdKHR
#endif
	md.size = p.alloc_size;                           // MUST equal VkMemoryRequirements.size
	md.flags = p.dedicated ? cudaExternalMemoryDedicated : 0;

	CUDA_CHECK(cudaImportExternalMemory(&s.extMem, &md));

	// OS handle can be closed after a successful import
#if defined(_WIN32)
	CloseHandle(p.memory_handle);
#else
	close(p.memory_fd);
#endif

	// 2.b) Map it as a mipmapped array (optimal tiling → array/texture access)
	cudaExternalMemoryMipmappedArrayDesc map{};
	map.offset = p.alloc_offset;                   // usually 0 for dedicated images
	map.numLevels = 1;
	map.extent = make_cudaExtent(p.width, p.height, 0);
	map.formatDesc = cudaCreateChannelDesc<uchar4>();  // BGRA8/RGBA8 as 4x8-bit

	CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&s.mmArr, s.extMem, &map));
	CUDA_CHECK(cudaGetMipmappedArrayLevel(&s.level0, s.mmArr, 0));

	// 2.c) (Optional) Create a texture object for convenient, cached reads
	cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = s.level0;
	cudaTextureDesc  td{}; td.readMode = cudaReadModeElementType; td.normalizedCoords = 0;
	td.filterMode = cudaFilterModePoint; td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
	CUDA_CHECK(cudaCreateTextureObject(&s.tex, &r, &td, nullptr));

	// 2.d) Import the semaphore Vulkan will signal (binary OR timeline)
	cudaExternalSemaphoreHandleDesc sd{};
#if defined(_WIN32)
	sd.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	sd.handle.win32.handle = p.semaphore_handle;      // HANDLE from vkGetSemaphoreWin32HandleKHR
#else
	sd.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	sd.handle.fd = p.semaphore_fd;                    // FD from vkGetSemaphoreFdKHR
#endif
	CUDA_CHECK(cudaImportExternalSemaphore(&s.extSem, &sd));

#if defined(_WIN32)
	CloseHandle(p.semaphore_handle);
#else
	close(p.semaphore_fd);
#endif
}

extern "C" const CudaImportedSlot* cuda_interop_get_slot(int index)
{
	if (index < 0 || index >= (int)g_slots.size()) return nullptr;
	return &g_slots[index];
}

// Convenience: import a whole array of packs
extern "C" void cuda_interop_import_all(const CudaImportPack* packs, int count)
{
	assert(packs && count > 0);
	g_slots.resize(count);
	for (int i = 0; i < count; ++i) {
		import_from_pack(packs[i], g_slots[i]);
	}
}

extern "C" bool cuda_interop_alloc_nv12_for_slot(int i, Nv12Surf* out)
{
	if (!out || i < 0 || i >= (int)g_slots.size()) return false;
	out->w = g_slots[i].width;
	out->h = g_slots[i].height;
	CUDA_CHECK(cudaMallocPitch((void**)&out->base, &out->pitch, out->w, out->h * 3 / 2));
	return true;
}

extern "C" bool cuda_interop_alloc_nv12_all()
{
	int count = g_slots.size();

	nv12.resize(count);
	for (int i = 0; i < count; ++i) 
	{
		if (!cuda_interop_alloc_nv12_for_slot(i, &nv12[i]))
		{
			return false;
		}
	}
	return true;
}

extern "C" void cuda_interop_free_nv12(Nv12Surf* s)
{
	if (s && s->base)
	{
		cudaFree(s->base); s->base = nullptr;
	}
}

extern "C" void cuda_interop_free_nv12_all()
{
	for (auto& s : nv12)
	{
		cuda_interop_free_nv12(&s);
	}
	nv12.clear();
}

extern "C" void cuda_interop_wait(int index, cudaStream_t stream)
{
	cudaExternalSemaphoreWaitParams waitParams{};
	waitParams.flags = 0;               // binary semaphore

	const CudaImportedSlot& s = g_slots[index];
	
	CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&s.extSem, &waitParams, 1, stream));
}

extern "C" void cuda_interop_destroy_all()
{
	for (auto& s : g_slots) {
		if (s.tex)    cudaDestroyTextureObject(s.tex);
		if (s.mmArr)  cudaFreeMipmappedArray(s.mmArr);
		if (s.extMem) cudaDestroyExternalMemory(s.extMem);
		if (s.extSem) cudaDestroyExternalSemaphore(s.extSem);
		s = {};
	}
	g_slots.clear();
}

extern "C" void cuda_interop_get_nv12_planes(const Nv12Surf* s,
	uint8_t** outY,
	uint8_t** outUV,
	size_t* outPitch)
{
	if (outY)   *outY = s ? s->base : nullptr;
	if (outUV)  *outUV = s ? (s->base ? s->base + s->pitch * s->h : nullptr) : nullptr;
	if (outPitch) *outPitch = s ? s->pitch : 0;
}


// ---------- BGRA (8-bit) -> NV12 (8-bit) ----------
// Assumes offscreen Vulkan RT is BGRA8. If you render RGBA8, swap r/b.

__device__ inline void rgb_to_yuv709_full(float r, float g, float b, float& Y, float& U, float& V) {
	Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;          // 0..1
	U = -0.1146f * r - 0.3854f * g + 0.5000f * b + 0.5f;   // center 0.5
	V = 0.5000f * r - 0.4542f * g - 0.0458f * b + 0.5f;
}

__global__ void bgra_to_nv12(cudaTextureObject_t tex,
	uint8_t* nv12Base, size_t pitch,
	int W, int H)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	// Read BGRA8 from Vulkan image (texture object, unnormalized coords)
	// +0.5f centers on texel
	const uchar4 bgra = tex2D<uchar4>(tex, x + 0.5f, y + 0.5f);
	const float b = bgra.x * (1.0f / 255.0f);
	const float g = bgra.y * (1.0f / 255.0f);
	const float r = bgra.z * (1.0f / 255.0f);

	float Yf, Uf, Vf;
	rgb_to_yuv709_full(r, g, b, Yf, Uf, Vf);

	// write Y
	uint8_t* yRow = nv12Base + (size_t)y * pitch;
	yRow[x] = (uint8_t)max(0.0f, min(255.0f, Yf * 255.0f));

	// write UV for top-left of each 2x2 block
	if (((x | y) & 1) == 0) {
		const int x1 = min(x + 1, W - 1), y1 = min(y + 1, H - 1);
		const uchar4 p01 = tex2D<uchar4>(tex, x1 + 0.5f, y + 0.5f);
		const uchar4 p10 = tex2D<uchar4>(tex, x + 0.5f, y1 + 0.5f);
		const uchar4 p11 = tex2D<uchar4>(tex, x1 + 0.5f, y1 + 0.5f);

		auto unpack = [](uchar4 q, float& r, float& g, float& b) {
			b = q.x * (1 / 255.f); g = q.y * (1 / 255.f); r = q.z * (1 / 255.f);
			};
		float r2, g2, b2, r3, g3, b3, r4, g4, b4;
		unpack(p01, r2, g2, b2); unpack(p10, r3, g3, b3); unpack(p11, r4, g4, b4);

		float Y2, U2, V2, Y3, U3, V3, Y4, U4, V4;
		rgb_to_yuv709_full(r2, g2, b2, Y2, U2, V2);
		rgb_to_yuv709_full(r3, g3, b3, Y3, U3, V3);
		rgb_to_yuv709_full(r4, g4, b4, Y4, U4, V4);

		const float U = (Uf + U2 + U3 + U4) * 0.25f;
		const float V = (Vf + V2 + V3 + V4) * 0.25f;

		uint8_t* uvBase = nv12Base + (size_t)H * pitch;
		uint8_t* uvRow = uvBase + (y / 2) * pitch;
		uvRow[x + 0] = (uint8_t)max(0.0f, min(255.0f, U * 255.0f));  // U
		uvRow[x + 1] = (uint8_t)max(0.0f, min(255.0f, V * 255.0f));  // V
	}
}

extern "C" void cuda_interop_convert_bgra_to_nv12(int slotIndex)
{
	Nv12Surf* dst = &nv12[slotIndex];

	// 1) Wait for Vulkan’s semaphore for this slot (binary)
	cuda_interop_wait(slotIndex, g_stream);

	// 2) Launch conversion
	const CudaImportedSlot& s = g_slots[slotIndex];
	dim3 block(16, 16);
	dim3 grid((s.width + block.x - 1) / block.x, (s.height + block.y - 1) / block.y);
	bgra_to_nv12<<<grid, block, 0, g_stream>>>(s.tex, dst->base, dst->pitch, s.width, s.height);
}


