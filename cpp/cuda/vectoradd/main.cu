#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// CUDA核函数，用于执行向量加法
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // 计算当前线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引在数组范围内
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
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
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d\n", i);
            break;
        }
    }
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