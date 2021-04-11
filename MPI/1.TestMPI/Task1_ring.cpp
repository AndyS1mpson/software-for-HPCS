#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "mpi.h"

#define BUFFSIZE 32

int main(int argc, char** argv)
{
	int rank, numtasks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Status status;

	int pr = (rank + 1) % numtasks;					// процесс, которой получает сообщения

	int ps = rank == 0 ? numtasks - 1 : rank - 1;	// процесс, от которого ожидается сообщение
	
	const char* msg = "Hello from process ";		
	char send_buff[BUFFSIZE];
	strcpy(send_buff, msg);
	int len = strlen(msg);
	send_buff[len] = '0' + rank;
	send_buff[len + 1] = '\0';
	char recv_buff[BUFFSIZE];

	MPI_Send(send_buff, strlen(send_buff) + 1, MPI_CHAR, pr, 0, MPI_COMM_WORLD);
	MPI_Recv(recv_buff, BUFFSIZE, MPI_CHAR, ps, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	printf("process %d received: %s\n", rank, recv_buff);
	MPI_Finalize();
}