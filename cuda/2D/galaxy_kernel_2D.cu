#include <glad/glad.h>
#include <GLFW/glfw3.h>

# include <cuda_runtime.h>
# include <cuda_gl_interop.h>
# include <vector_types.h>

#define BSIZE_X  16
#define BSIZE_Y  16
#define BSIZE (BSIZE_X * BSIZE_Y)

#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f						// 0.5f

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;
        
   	float dist = sqrtf(distSqr);
   	float distCube = dist * dist * dist;

	if (distCube < 1.0f) return ai;     // avoid singularity
	
    float s = bi.w / distCube;
    //float s = 1.0f / distCube;
    
    ai.x += r.x * s * ep;
    ai.y += r.y * s * ep;
    ai.z += r.z * s * ep;

    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 acc)
{
	extern __shared__ float4 shPosition[];
	
	#pragma unroll 8
	for (unsigned int i = 0; i < BSIZE; i++){
		acc = bodyBodyInteraction(myPosition, shPosition[i], acc);
    }	
	return acc;
}

__global__ void galaxyKernel2D(float4* pos, float4* pdata,int width, int height, float step, int apprx, int offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int pLoc = y * width + x;
    int N = width * height;
    int vLoc = N + pLoc;
    int start = (N / apprx) * offset;

    extern __shared__ float4 shPosition[];

    float4 myPosition = pdata[pLoc];
    float4 myVelocity = pdata[vLoc];

    float3 acc = {0.0f, 0.0f, 0.0f};

    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int loop = (N / apprx) / BSIZE;
    for (int i = 0; i < loop; i++) {
        int globalIdx = start + i * BSIZE + localIdx;
        shPosition[localIdx] = pdata[globalIdx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }

    myVelocity.x += acc.x * step;
    myVelocity.y += acc.y * step;
    myVelocity.z += acc.z * step;

    myVelocity.x *= damping;
    myVelocity.y *= damping;
    myVelocity.z *= damping;

    myPosition.x += myVelocity.x * step;
    myPosition.y += myVelocity.y * step;
    myPosition.z += myVelocity.z * step;

    pdata[pLoc] = myPosition;
    pdata[vLoc] = myVelocity;

    pos[2 * pLoc] = make_float4(myPosition.x, myPosition.y, myPosition.z, 1.0f);
    pos[2 * pLoc + 1] = make_float4(myVelocity.x, myVelocity.y, myVelocity.z, 1.0f);
}

extern "C" void cudaComputeGalaxy2D(void* dptr, float4* pdata,int width, int height, float step, int apprx, int offset) {
    dim3 block(BSIZE_X, BSIZE_Y);
    dim3 grid((width + BSIZE_X - 1) / BSIZE_X, (height + BSIZE_Y - 1) / BSIZE_Y);
    float4* out = reinterpret_cast<float4*>(dptr);
    size_t sharedMem = BSIZE * sizeof(float4);
    galaxyKernel2D<<<grid, block,sharedMem>>>(out, pdata, width, height, step, apprx, offset);
}
