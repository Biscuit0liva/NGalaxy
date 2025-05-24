#define BSIZE_X 16
#define BSIZE_Y 16
#define softeningSquared 0.01f
#define damping 1.0f
#define ep 0.67f

inline float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r = (float3)(bj.x - bi.x,
                        bj.y - bi.y,
                        bj.z - bi.z);
    float distSqr = dot(r, r) + softeningSquared;
    float dist = sqrt(distSqr);
    float distCube = dist * dist * dist;
    if (distCube < 1.0f)
        return ai;
    float s = bi.w / distCube;
    ai.x += r.x * s * ep;
    ai.y += r.y * s * ep;
    ai.z += r.z * s * ep;
    return ai;
}

inline float3 tile_calculation(float4 myPosition,
                               float3 acc,
                               __local float4* shPosition,
                               int tileSize)
{
    // cada work-item recorre el tile en local memory
    for (int i = 0; i < tileSize; i++) {
        acc = bodyBodyInteraction(myPosition, shPosition[i], acc);
    }
    return acc;
}

__kernel void galaxyKernel2D(__global float4* pos,
                             __global float4* pdata,
                             const int N,
                             const float step,
                             const int apprx,
                             const int offset)
{
    // Obtener coordenadas 2D
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    
    // Calcular índice lineal desde coordenadas 2D
    int gid = gy * get_global_size(0) + gx;
    int lid = ly * get_local_size(0) + lx;
    
    if (gid >= N) return;
    
    int vLoc = N + gid;
    int start = (N / apprx) * offset;
    
    // tile en memoria local - ahora usando el tamaño del work-group 2D
    int localWorkGroupSize = get_local_size(0) * get_local_size(1);
    __local float4 shPosition[BSIZE_X * BSIZE_Y];
    
    float4 myPosition = pdata[gid];
    float4 myVelocity = pdata[vLoc];
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);
    
    int loop = (N / apprx) / localWorkGroupSize;
    
    for (int t = 0; t < loop; t++) {
        // cada hilo carga una posicion a local usando su ID local
        int loadIndex = start + t * localWorkGroupSize + lid;
        if (loadIndex < N) {
            shPosition[lid] = pdata[loadIndex];
        } else {
            // Si estamos fuera de rango, llenar con datos neutros
            shPosition[lid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Calcular cuántas partículas válidas hay en este tile
        int validParticles = min(localWorkGroupSize, N - (start + t * localWorkGroupSize));
        acc = tile_calculation(myPosition, acc, shPosition, validParticles);
        
        barrier(CLK_LOCAL_MEM_FENCE);
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
    pdata[gid] = myPosition;
    pdata[vLoc] = myVelocity;
    
    // actualiza buffer de visualizacion interop
    pos[2 * gid] = (float4)(myPosition.x, myPosition.y, myPosition.z, 1.0f);
    pos[2 * gid + 1] = (float4)(myVelocity.x, myVelocity.y, myVelocity.z, 1.0f);
}
