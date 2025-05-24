// galaxy_kernel.cl

#define BSIZE 256
#define softeningSquared 0.01f
#define damping 1.0f
#define ep 0.67f

inline float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r = (float3)(bj.x - bi.x,
                        bj.y - bi.y,
                        bj.z - bi.z);
    float distSqr  = dot(r, r) + softeningSquared;
    float  dist     = sqrt(distSqr);
    float  distCube = dist * dist * dist;

    if (distCube < 1.0f)
        return ai;

    float s = bi.w / distCube;
    ai.x += r.x * s * ep;
    ai.y += r.y * s * ep;
    ai.z += r.z * s * ep;
    return ai;
}


__kernel void galaxyKernel(__global float4* pos,
                           __global float4* pdata,
                           const int     N,
                           const float   step,
                           const int     apprx,
                           const int     offset)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    int vLoc  = N + gid;
    int start = (N / apprx) * offset;

    float4 myPosition = pdata[gid];
    float4 myVelocity = pdata[vLoc];
    float3 acc        = (float3)(0.0f, 0.0f, 0.0f);

    int loop = (N / apprx) / BSIZE;

    // por cada tile, recorremos BSIZE elementos directamente en global
    for (int t = 0; t < loop; t++) {
        int base = start + t * BSIZE;
        // cada hilo recorre todo el tile leyendo de global
        for (int i = 0; i < BSIZE; i++) {
            float4 bj = pdata[ base + i ];
            acc = bodyBodyInteraction(myPosition, bj, acc);
        }
    }

    // actualiza velocidad y posicion 
    myVelocity.x += acc.x * step;
    myVelocity.y += acc.y * step;
    myVelocity.z += acc.z * step;

    myVelocity.x *= damping;
    myVelocity.y *= damping;
    myVelocity.z *= damping;

    myPosition.x += myVelocity.x * step;
    myPosition.y += myVelocity.y * step;
    myPosition.z += myVelocity.z * step;

    // escribe de vuelta
    pdata[gid]      = myPosition;
    pdata[vLoc]     = myVelocity;

    // actualiza buffer de visualizacion interop
    pos[2 * gid]     = (float4)(myPosition.x, myPosition.y, myPosition.z, 1.0f);
    pos[2 * gid + 1] = (float4)(myVelocity.x, myVelocity.y, myVelocity.z, 1.0f);
}
