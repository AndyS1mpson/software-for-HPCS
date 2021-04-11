#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "mpi.h"

#define BUFFSIZE 32
#define MAXTASKS 8


int main(int argc, char** argv)
{
	int rank, numtasks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Status status;

	int pr = 0;
	const char* msg = "Hello from process ";
	char send_buff[BUFFSIZE];
	strcpy(send_buff, msg);
	int len = strlen(msg);
	send_buff[len] = '0' + rank;
	send_buff[len + 1] = '\0';

	if (rank == 0)
	{
		char recv_buff[MAXTASKS - 1][BUFFSIZE];
		for (int i = 0; i < numtasks - 1; ++i)
		{
			MPI_Recv(recv_buff[i], BUFFSIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Send(send_buff, strlen(send_buff) + 1, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			printf("The process received: %s\n", recv_buff[i]);
		}
	}
	else {
		char recv_from_master[BUFFSIZE];
		MPI_Send(send_buff, strlen(send_buff) + 1, MPI_CHAR, pr, 0, MPI_COMM_WORLD);
		MPI_Recv(recv_from_master, BUFFSIZE, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		printf("Received from main process: %s\n", recv_from_master);
	}

	MPI_Finalize();
}