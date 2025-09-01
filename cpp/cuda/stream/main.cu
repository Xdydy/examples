#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>


// CUDA核函数，用于执行向量加法
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // 计算当前线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引在数组范围内
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void vectorAddBaseline(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 100000000; // 向量大小
    size_t size = n * sizeof(int);
    
    // 在主机上分配内存并初始化数据
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 在设备上分配内存
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 定义线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动核函数
    cudaStream_t stream;
    stream = nullptr; // 使用默认流
    std::cout << "Start function vectorAdd" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    stream = nullptr;
    std::cout << "Start function vectorAddBaseline" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    vectorAddBaseline<<<1, 1, 0, stream>>>(d_a, d_b, d_c, n);
    cudaStreamSynchronize(stream);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    printf("Vector addition completed successfully!\n");

    
    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}