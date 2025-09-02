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

int main() {
    int N = 512;
    size_t size = N * N * sizeof(int);

    int ha[N][N];
    int hb[N][N];
    int hc[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ha[i][j] = i + j;
            hb[i][j] = i - j;
            hc[i][j] = 0;
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
    gemm<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(hc, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // check result
    int res[N][N];
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0 ;
            for (int k = 0; k < N; k++) {
                sum += ha[i][k] * hb[k][j];
            }
            res[i][j] = sum;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "CPU Time taken: " << duration.count() << " seconds" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (res[i][j] != hc[i][j]) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " << res[i][j] << " != " << hc[i][j] << std::endl;
                return -1;
            }
        }
    }

    return 0;
}