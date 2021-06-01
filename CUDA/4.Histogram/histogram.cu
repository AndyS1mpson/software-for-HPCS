#include "cuda.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <fstream>
#include <cstring>
#include <ctime>

using namespace std;

const int countLetter = 26;
const int lenLine = 1024;


__global__ void SumLetterCuda(int* gist, char* line, char* letter)
{
	int i = blockIdx.x;  //countLetter = 26
	int j = threadIdx.x; // lenLine = 1024
	if (letter[i] == line[j])
		atomicAdd(&gist[i], 1);
}

void SumLetter(int* gist, char* line, char* letter)
{
	for (int i = 0; i < lenLine; i++) // 1024
	{
		for (int j = 0; j < countLetter; j++) //26
		{
			if (line[i] == letter[j])
				gist[j]++;
		}
	}
}

void PrintMas(int* mas, int N)
{
	for (int i = 0; i < N; i++) {
		cout << mas[i] << " ";
	}
	cout << endl;
}

void PrintMasChar(char* mas, int N)
{
	for (int i = 0; i < N; i++) {
		cout << mas[i] << " ";
	}
	cout << endl;
}


int main()
{
	float Time = 0;

	int* gist = new int[countLetter]; //26
	for (int i = 0; i < countLetter; i++)
		gist[i] = 0;
	char letter[countLetter] = { 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z' };
	char buff[lenLine]; //1024

	ifstream fin("text.txt");
	fin.getline(buff, lenLine);
	fin.close();

	Time = clock();
	SumLetter(gist, buff, letter);
	cout << "Время выполнения на CPU " << Time << endl;

	cout << "Результат работы:" << endl;
	PrintMasChar(letter, countLetter);
	PrintMas(gist, countLetter);

	int* cudaGist;
	char* cudaLetter;
	char* cudaBuff;
	
	// Переменные для измерения времени выполнения CUDA.
	Time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Выделение памяти
	cudaMalloc((void**)&cudaGist, sizeof(int) * countLetter);
	cudaMalloc((void**)&cudaLetter, sizeof(char) * countLetter);
	cudaMalloc((void**)&cudaBuff, sizeof(char) * lenLine);

	cudaMemcpy(cudaGist, gist, sizeof(int) * countLetter, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaLetter, letter, sizeof(char) * countLetter, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaBuff, buff, sizeof(char) * lenLine, cudaMemcpyHostToDevice);

	dim3 blocks = countLetter;
	dim3 threads = lenLine;

	// Запуск таймера
	cudaEventRecord(start);
	
	SumLetterCuda << < blocks, threads >> > (cudaGist, cudaBuff, cudaLetter);
	cudaMemcpy(gist, cudaGist, sizeof(int) * countLetter, cudaMemcpyDeviceToHost);

	// Остановка таймера
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&Time, start, end);
	cout << "Время работы на GPU " << Time << endl;
	cout << "Результат работы:" << endl;
	PrintMasChar(letter, countLetter);
	PrintMas(gist, countLetter);

	cudaFree(cudaGist);
	cudaFree(cudaBuff);
	cudaFree(cudaLetter);

	return 0;
}