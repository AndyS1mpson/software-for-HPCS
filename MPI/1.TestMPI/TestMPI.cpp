#include "mpi.h"
#include <stdio.h>


int main(int argc, char** argv)
{
	int size, rank;
	// Начало параллельной части программы
	MPI_Init(&argc, &argv);
	
	// Функция, позволяющая узнать общее число процессов в глобальном коммуникаторе
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Функция, позволяющая узнать номер процесса в области связи
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Вывод 
	printf("Total number of processes in the communicator: %d, current procces number: %d\n", size, rank);

	//Конец параллельной части программы
	return 0;
}
