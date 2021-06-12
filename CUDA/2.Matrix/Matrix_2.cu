#include "stdio.h"
#include <stdlib.h>

__global__
void MatAdd(int* A, int* B, int* C, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;

    C[idx] = A[idx] + B[idx];
}

void GenerateMatrix(int* m, int size)
{
    for (int i=0; i<size; ++i)
        m[i] = rand()% 10;
}

int main( void ) {

    // size of matrix
    int nx = 4; // столбцы
    int ny = 5; // строки
    int size = nx*ny;
    size_t sizeBytes = size * sizeof(int);

    int* h_A = (int*)malloc(sizeBytes);
    int* h_B = (int*)malloc(sizeBytes);
    int* h_C = (int*)malloc(sizeBytes);

    GenerateMatrix(h_A, size);
    GenerateMatrix(h_B, size);

    int* d_A;
    int* d_B;
    int* d_C;
    cudaMalloc((void**)&d_A, sizeBytes);
    cudaMalloc((void**)&d_B, sizeBytes);
    cudaMalloc((void**)&d_C, sizeBytes);

    cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice);

    int BlockPerGrid = ny; // строки
    int ThreadsPerBlock = nx; // столбцы

    MatAdd<<< BlockPerGrid, ThreadsPerBlock >>>(d_A, d_B, d_C, nx, ny);
    cudaMemcpy(h_C, d_C, sizeBytes, cudaMemcpyDeviceToHost);

    // display the results
    for (int i=0; i<size; i++) {
        printf( "%d + %d = %d    ", h_A[i], h_B[i], h_C[i] );
        if (i%size == 0) printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
