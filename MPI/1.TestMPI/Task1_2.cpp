#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mpi.h"
#include<string>  
#include <iostream>
#define BUFFSIZE 32
#define MAXTASKSAMOUNT 8


void create_message(char message[BUFFSIZE], int rank)
{
	const char* m = "Hello from process ";
	strcpy(message, m);
	int len = strlen(m);

}

int main(int argc, char** argv)
{
	int numtasks, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Status status;
	MPI_Request request;
	// буферы отправляемых сообщений
	char send_buff[BUFFSIZE];
	// буфер полученных сообщений
	char recv_buff[MAXTASKSAMOUNT][BUFFSIZE];
	create_message(send_buff, rank);
	// Отправка сообщения другим процессам
	for (int i = 0; i < numtasks; ++i)
	{
		MPI_Send(send_buff, strlen(send_buff) + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
	}
	// Получение сообщений от других процессов
	for (int i = 0; i < numtasks; ++i)
	{
		MPI_Recv(recv_buff[i], BUFFSIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		printf("process %d received: %s\n", rank, recv_buff[i]);

	}

	MPI_Finalize();
}