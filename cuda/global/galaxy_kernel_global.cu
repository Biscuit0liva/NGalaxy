#include <glad/glad.h>
#include <GLFW/glfw3.h>

# include <cuda_runtime.h>
# include <cuda_gl_interop.h>
# include <vector_types.h>


#define BSIZE 256
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f						// 0.5f

__global__ void galaxyKernelGlobal(float4* pos, float4* pdata, int N, float step, int apprx, int offset) {
    unsigned int pLoc = blockIdx.x * blockDim.x + threadIdx.x;
    if (pLoc >= N) return;

    unsigned int vLoc = N + pLoc;
    unsigned int start = (N / apprx) * offset;


    float4 myPosition = pdata[pLoc];
    float4 myVelocity = pdata[vLoc];

    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 r;
    float distSqr, distCube, s;

    for(int i=0; i<N; i++){
        r.x = pdata[i].x - myPosition.x;
        r.y = pdata[i].y - myPosition.y;
        r.z = pdata[i].z - myPosition.z;

        distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
        distSqr += softeningSquared;

        float dist = sqrtf(distSqr);
        distCube = dist * dist * dist;
        if (distCube < 1.0f) continue;

        s = pdata[i].w / distCube;

        acc.x += r.x * s * ep;
        acc.y += r.y * s * ep;
        acc.z += r.z * s * ep;
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
    
    __syncthreads();

    pdata[pLoc] = myPosition;
    pdata[vLoc] = myVelocity;

    pos[2 * pLoc] = make_float4(myPosition.x, myPosition.y, myPosition.z, 1.0f);
    pos[2 * pLoc + 1] = make_float4(myVelocity.x, myVelocity.y, myVelocity.z, 1.0f);
}

extern "C" void cudaComputeGalaxyGlobal(void* dptr, float4* pdata, int N, float step, int apprx, int offset) {
    float4* out = reinterpret_cast<float4*>(dptr);
    int blockSize = BSIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    galaxyKernelGlobal<<<numBlocks, blockSize, BSIZE * sizeof(float4)>>>(out, pdata, N, step, apprx, offset);
}
