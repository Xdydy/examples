#include <bits/stdc++.h>

__device__ int getElement(int *a, int i, int j, int N) {
    return a[i*N+j];
}

__global__ void gemm(int *a, int *b, int *c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int tileA[16][16];
    __shared__ int tileB[16][16];

    int sum = 0;
    for (int t = 0; t < (N + 15) / 16; t++) {
        tileA[threadIdx.x][threadIdx.y] = getElement(a, row, t * 16 + threadIdx.y, N);
        tileB[threadIdx.x][threadIdx.y] = getElement(b, t * 16 + threadIdx.x, col, N);
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            sum += tileA[threadIdx.x][k] * tileB[k][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        c[row * N + col] = sum;
    }
}

__global__ void gemmBaseline(int *a, int *b, int *c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = 0 ;
    for (int i = 0; i < N; i++) {
        sum += getElement(a, row, i, N) * getElement(b, i, col, N);
    }
    c[row * N + col] = sum;
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(int);

    int *ha = (int*)malloc(size);
    int *hb = (int*)malloc(size);
    int *hc = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ha[i*N+j] = 1;
            hb[i*N+j] = 1;
            hc[i*N+j] = 0;
        }
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    gemmBaseline<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(hc, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Baseline Time taken: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    gemm<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(hc, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "gemm Time taken: " << duration.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // check result
    int *res = (int*)malloc(size);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0 ;
            for (int k = 0; k < N; k++) {
                sum += ha[i*N+k] * hb[k*N+j];
            }
            res[i*N+j] = sum;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "CPU Time taken: " << duration.count() << " seconds" << std::endl;
    for (int i = 0; i < size; i++) {
        if (res[i] != hc[i]) {
            std::cout << "Mismatch at (" << i << "): " << res[i] << " != " << hc[i] << std::endl;
            return -1;
        }
    }

    return 0;
}