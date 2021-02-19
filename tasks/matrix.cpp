#include <iostream>
#include <math.h> 
#include <omp.h>
#include <stdio.h>
using namespace std;

int main()
{
	// Размерность матрицы 
	int n = 3;
	int m = 3;
	// Создаем матрицу

	double** matrix = new double* [10];
	for (int i = 0; i < n; i++)
	{
		matrix[i] = new double[m];
	}

	// Заполняем матрицу 
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			cin >> matrix[i][j];
		}
	}




	// Наше искомое максимальное значение среди минимальных значений строк матрицы 
	double max = 0;

	// Массив минимальных элементов 
	double* min_values = new double[n] { 0 };

	// Поиск всех минимальных элементов
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if ((matrix[i][j] < min_values[i]) || (min_values[i] == 0))
			{
				min_values[i] = matrix[i][j];
			}
		}
	}

	// Поиск максимального среди минимальных
	for (int j = 0; j < n; j++)
	{
		if ((min_values[j] > max) || (max == 0))
		{
			max = min_values[j];
		}
	}

	cout << max << "\n";


}

