#include <bits/stdc++.h>
#include <cuda_runtime.h>

template<int NUM_THREADS = 256>
__global__ void sum(int *a, int *output, int N) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int cache[NUM_THREADS];
    cache[tid] = (index < N) ? a[index] : 0;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid % (2 * offset) == 0) {
            cache[tid] += cache[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, cache[0]);
    }
}

int main() {
    int N = 1 << 20;
    int *a, *d_a;
    a = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) a[i] = 1;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    int *d_output;
    cudaMalloc(&d_output, sizeof(int));
    auto start = std::chrono::high_resolution_clock::now();
    sum<<<(N + 255) / 256, 256>>>(d_a, d_output, N);
    int output;
    cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Result: " << (1 << 20) << std::endl;
    std::cout << "Sum: " << output << std::endl;
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
    cudaFree(d_a);
    cudaFree(d_output);
    free(a);
    return 0;
}