#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__
void MatAdd1D(int* A, int* B, int* C, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;

    C[idx] = A[idx] + B[idx];
}

__global__
void MatAdd2D(int* A, int* B, int* C, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

__global__
void MatAdd1D1D(int* A, int* B, int* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}


__global__
void MatAdd2D1D(int* A, int* B, int* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    unsigned int idx = iy*nx + ix;
    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

void sumMatrixOnHost (int *A, int *B, int *C, const int nx, const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

void GenerateMatrix(int* m, int size)
{
    for (int i=0; i<size; ++i)
        m[i] = rand()% 10;
}

int main( void ) {

    double time1, time2, time3, time4;

    // size of matrix
    unsigned int nx = 1<<10; // столбцы
    unsigned int ny = 1<<10; // строки
    int size = nx*ny;
    size_t sizeBytes = size * sizeof(int);

    int* h_A = (int*)malloc(sizeBytes);
    int* h_B = (int*)malloc(sizeBytes);
    int* h_C = (int*)malloc(sizeBytes);
    int* cpu_C = (int*)malloc(sizeBytes);

    GenerateMatrix(h_A, size);
    GenerateMatrix(h_B, size);
    sumMatrixOnHost(h_A, h_B, cpu_C, nx, ny);

    int* d_A;
    int* d_B;
    cudaMalloc((void**)&d_A, sizeBytes);
    cudaMalloc((void**)&d_B, sizeBytes);

    cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice);

    printf("Started succesfylly\n");

    /* Варивант с 1D */

    int* d_C1D;
    cudaMalloc((void**)&d_C1D, sizeBytes);

    int BlockPerGrid = ny; // строки
    int ThreadsPerBlock = nx; // столбцы

    cudaDeviceSynchronize();
    time1 = cpuSecond();
    MatAdd1D<<< BlockPerGrid, ThreadsPerBlock >>>(d_A, d_B, d_C1D, nx, ny);
    cudaDeviceSynchronize();
    time1 = cpuSecond() - time1;
    printf("MattAdd1D <<<(%d, %d)>>> elapsed %f ms\n", BlockPerGrid,
           ThreadsPerBlock, time1);

    cudaMemcpy(h_C, d_C1D, sizeBytes, cudaMemcpyDeviceToHost);
    checkResult(cpu_C, h_C, size);
    cudaFree(d_C1D);

    /* Вариант с 2D */

    int* d_C2D;
    cudaMalloc((void**)&d_C2D, sizeBytes);

    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaDeviceSynchronize();
    time2 = cpuSecond();
    MatAdd2D<<<grid, block>>>(d_A, d_B, d_C2D, nx, ny);
    cudaDeviceSynchronize();
    time2 = cpuSecond() - time2;
    printf("MattAdd2D <<<(%d, %d), (%d, %d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, time2);

    cudaMemcpy(h_C, d_C1D, sizeBytes, cudaMemcpyDeviceToHost);
    checkResult(cpu_C, h_C, size);
    cudaFree(d_C1D);

    /* Вариант с 1D-сеткой и 1D-блоками */

    int* d_C1D1D;
    cudaMalloc((void**)&d_C1D1D, sizeBytes);

    block = dim3{128,1};
    grid = dim3{(nx+block.x-1)/block.x,1};

    cudaDeviceSynchronize();
    time3 = cpuSecond();
    MatAdd1D1D <<<grid, block>>> (d_A, d_B, d_C1D1D, nx, ny);
    cudaDeviceSynchronize();
    time3 = cpuSecond() - time3;
    printf("MatAdd1D1D <<<(%d, %d), (%d, %d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, time3);
    cudaMemcpy(h_C, d_C1D1D, sizeBytes, cudaMemcpyDeviceToHost);
    checkResult(cpu_C, h_C, size);
    cudaFree(d_C1D1D);


    /* Вариант с 2D-сеткой и 1D-блоками */
    int* d_C2D1D;
    cudaMalloc((void**)&d_C2D1D, sizeBytes);

    block = dim3{256};
    grid = dim3{(nx + block.x - 1) / block.x,ny};

    cudaDeviceSynchronize();
    time4 = cpuSecond();
    MatAdd2D1D <<<grid, block>>> (d_A, d_B, d_C2D1D, nx, ny);
    cudaDeviceSynchronize();
    time4 = cpuSecond() - time4;
    printf("MatAdd2D1D <<<(%d, %d), (%d, %d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, time4);
    cudaMemcpy(h_C, d_C2D1D, sizeBytes, cudaMemcpyDeviceToHost);
    checkResult(cpu_C, h_C, size);
    cudaFree(d_C2D1D);


    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
