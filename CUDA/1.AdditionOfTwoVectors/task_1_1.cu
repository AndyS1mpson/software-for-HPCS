#include "stdio.h"
#include <stdlib.h>

#define N   10

__global__
void add( int *a, int *b, int *c ) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main( void ) {
    size_t size = N* sizeof(int);

    int* h_a = (int*)malloc(size);
    int* h_b = (int*)malloc(size);
    int* h_c = (int*)malloc(size);
    for (int i=0; i<N; i++) {
        h_a[i] = -i;
        h_b[i] = i * i;
    }

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1;
    int blockPerGrid = 1;

    add <<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", h_a[i], h_b[i], h_c[i] );
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
