#include "cuda.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>

using namespace std;
const int Radius = 1000;
const int N = 100000;

__global__ void PiCuda(double* points, int* count)
{
	//__shared__ int cCount[N];
	//for (int i = 0; i < N; i++)
		//cCount[i] = 0;
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	if (points[tx * 2] * points[tx * 2] + points[tx * 2 + 1] * points[tx * 2 + 1] < Radius * Radius)
		atomicAdd(count, 1);
	//cCount[tx]++;
//__syncthreads();
//count[0] += cCount[tx];
}

int Pi(double* points)
{
	int count = 0;
	for (int i = 0; i < N; i++)
	{
		if (points[i * 2] * points[i * 2] + points[i * 2 + 1] * points[i * 2 + 1] < Radius * Radius)
			count++;
	}

	return count;
}



void CreatePoints(double* Points)
{
	srand(time(0));
	for (int i = 0; i < N; i++)
	{
		Points[i * 2] = rand() % Radius;
		Points[i * 2 + 1] = rand() % Radius;
	}
}

int main()
{
	//Переменные для измерения времени выполнения CUDA.
	cudaEvent_t start, end;
	float Time = 0;

	double points[N * 2];
	CreatePoints(points);
	int countPointsInCircle = 0;

	double* cudaPoints;
	int* cudaCountPointsInCircle;
	cudaMalloc((void**)&cudaPoints, sizeof(double) * N * 2);
	cudaMalloc((void**)&cudaCountPointsInCircle, sizeof(int));

	cudaMemcpy(cudaPoints, &points, sizeof(double) * N * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaCountPointsInCircle, &countPointsInCircle, sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	// Запуск таймера
	cudaEventRecord(start);
	float t_start = clock();
	
	countPointsInCircle = Pi(points);
	float t_end = clock();
	cout << "Время работы последовательной версии:" << t_end - t_start << endl;
	// Остановка таймера
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&Time, start, end);
	
	cout << "PI = " << 4 * double(countPointsInCircle) / N << endl;

	dim3 blocks = 100;
	dim3 threads = 1000;
	// Запуск таймера
	cudaEventRecord(start);
	PiCuda << < blocks, threads >> > (cudaPoints, cudaCountPointsInCircle);
	// Остановка таймера
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&Time, start, end);
	cout << "Время работы на GPU:" << Time << endl;
	cudaDeviceSynchronize();
	cudaMemcpy(&countPointsInCircle, cudaCountPointsInCircle, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "PI = " << 4 * float(countPointsInCircle) / float(N) << endl;

	cudaFree(cudaPoints);

	return 0;
}