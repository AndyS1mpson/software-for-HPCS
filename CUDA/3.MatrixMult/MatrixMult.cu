#include "cuda.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int N = 8;
const int block = 4;

__global__ void MultMatrCuda(int* res, int* mas1, int* mas2)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = N * block * by;
	int aEnd = aBegin + N - 1;
	int aStep = block;

	int bBegin = block * bx;
	int bStep = block * N;

	int sum = 0;

	for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep)
	{
		__shared__ int as[block][block];
		__shared__ int bs[block][block];

		as[ty][tx] = mas1[a + N * ty + tx];
		bs[ty][tx] = mas2[b + N * ty + tx];

		__syncthreads();

		for (int k = 0; k < block; k++)
		{
			sum += as[ty][k] * bs[k][tx];

			__syncthreads();
		}

		int c = N * block * by + block * bx;

		res[c + N * ty + tx] = sum;
	}

}

void MultMatr(int* res, int* mas1, int* mas2)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++)
				res[i * N + j] += mas1[i * N + k] * mas2[k * N + j];
		}
	}
}

void PrintMas(int* mas)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			cout << mas[i * N + j] << " ";
		cout << endl;
	}
	cout << endl;
}

int main()
{
	int mas1[N * N] = { 1,1,1,1,1,1,1,1,
						2,2,2,2,2,2,2,2,
						3,3,3,3,3,3,3,3,
						4,4,4,4,4,4,4,4,
						5,5,5,5,5,5,5,5,
						6,6,6,6,6,6,6,6,
						7,7,7,7,7,7,7,7,
						8,8,8,8,8,8,8,8 };
	int mas2[N * N] = { 1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1,
						1,1,1,1,1,1,1,1 };
	
	
	int* res = new int[N * N];

	int* cudaMas1;
	int* cudaMas2;
	int* cudaRes;

	// Переменные для измерения времени выполнения CUDA.
	cudaEvent_t start, end;
	float Time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	// Выделение памяти
	cudaMalloc((void**)&cudaMas1, sizeof(int) * N * N);
	cudaMalloc((void**)&cudaMas2, sizeof(int) * N * N);
	cudaMalloc((void**)&cudaRes, sizeof(int) * N * N);

	cudaMemcpy(cudaMas1, mas1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaMas2, mas2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaRes, res, sizeof(int) * N * N, cudaMemcpyHostToDevice);

	// Вывод матриц 
	PrintMas(mas1);
	PrintMas(mas2);
	
	// Засекаем время
	cudaEventRecord(start);
	
	MultMatr(res, mas1, mas2);
	// Останавливаем таймер
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&Time, start, end);
	cout << "Время работы " << Time << endl;

	cout << "Произведение " << endl;
	PrintMas(res);

	dim3 threads(block, block);
	dim3 blocks(N / threads.x, N / threads.y);
	
	// Засекаем время
	cudaEventRecord(start, 0);
	
	MultMatrCuda << < blocks, threads >> > (cudaRes, cudaMas1, cudaMas2);
	// Останавливаем таймер
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&Time, start, end);
	cout << "Время работы " << Time << endl;
	cudaMemcpy(res, cudaRes, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

	cout << "Произведение" << endl;
	PrintMas(res);

	cudaFree(cudaMas1);
	cudaFree(cudaMas2);
	cudaFree(cudaRes);

	return 0;
}