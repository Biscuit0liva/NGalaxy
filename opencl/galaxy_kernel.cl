__kernel void vector_addition(__global const float *a, 
                              __global const float *b, 
                              __global float *result, 
                              const unsigned int n) {
    // Get the index of the current element
    int id = get_global_id(0);

    // Perform the addition if within bounds
    if (id < n) {
        result[id] = a[id] + b[id];
    }
}