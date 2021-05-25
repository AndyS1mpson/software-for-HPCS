#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <mpi.h>
using namespace std;

const int N=100;
double cA[N][N+1];
double A[N][N + 1];
double x[N]={0};

void generate_matrix()
{
    int i, j;
    srand(time(0));
    for (i = 0; i < N; ++i)
        for (j = 0; j < N + 1; ++j)
            A[i][j] = cA[i][j] = rand() % 10 + 1;
}

void output_matrix()
{
    int i, j;
    cout.precision(3);
    for (i = 0; i < N + 1; ++i) cout << "------";
    cout << fixed << "-----" << endl;
    for (i = 0; i < N; ++i)
    {
        cout << "| ";
        for (j = 0; j < N; ++j)
            cout << setw(5) << cA[i][j] << " ";
        cout << "| " << setw(5) << cA[i][j];
        cout << " |  x[" << setw(2) << i << "] = " << setw(5) << x[i] << endl;
    }
    for (i = 0; i < N + 1; ++i) cout << "------";
    cout << "-----" << endl;
}

void test()
{
    int i, j;
    double diff, sum;
    cout.precision(10);

    for (i = 0; i < N; ++i)
    {
        sum = 0;
        for (j = 0; j < N; ++j)
            sum += x[j] * A[i][j];
        diff = sum + A[i][N];
        if (diff > 0.0001 || diff < -0.0001)
            cout << "ERROR! " << sum << " ~ " << A[i][N] << ", diff:" << diff << endl;
        if (N < 50)
        {
            cout << setw(4) << sum << " ~ " << setw(4) << A[i][N];
            cout << ", diff:" << setw(4) << fixed << diff << endl;
        }
    }
}


int main(int argc, char* argv[])
{
    int ProcRank, ProcNum, RankFrom;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    int irow,jrow, j;
    double coef, les;
    double t1;


    if (ProcRank==0)
    {
        generate_matrix();
        t1 = MPI_Wtime();
    }
    MPI_Bcast(cA, N*(N+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (irow=0;irow<N-1;++irow)
    {
        RankFrom = (N - 1 - irow) % ProcNum;
        MPI_Bcast(&cA[irow], N+1, MPI_DOUBLE, RankFrom, MPI_COMM_WORLD);
        coef = -1.0/cA[irow][irow];
        for (jrow=N-1-ProcRank; jrow>=irow+1; jrow -= ProcNum)
        {
            les = cA[jrow][irow] * coef;
            for(j=irow;j<N+1;++j)
                cA[jrow][j] += les * cA[irow][j];
        }
    }

    if (ProcRank==0)
    {
        for(irow=N-1; irow>=0; --irow)
        {
            x[irow] = -cA[irow][N]/cA[irow][irow];
            for (jrow=0; jrow<irow; ++jrow)
            {
                cA[jrow][N] += x[irow] * cA[jrow][irow];
                cA[jrow][irow]=0;
            }
        }

        cout << "Size: " << N << ", time: " <<  (MPI_Wtime()-t1)*1000 << "ms" <<endl;
    }

    MPI_Finalize();
}

