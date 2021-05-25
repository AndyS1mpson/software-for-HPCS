#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <mpi.h>
using namespace std;

const int N=100;
double cA[N][N+1];
double x[N]={0};
double A[N][N + 1];

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
    for (i = 0; i < N + 1; ++i)cout << "------";
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


int main()
{
    int i,j, k;
    double coef, les;
    double t1;

    generate_matrix();

    t1 = MPI_Wtime();
    for (i=0; i<N-1; ++i)
    {
        coef = -1.0/cA[i][i];
        for (j=i+1; j<N; ++j)
        {
            les = cA[j][i] * coef;
            for(k=i;k<N+1;++k)
                cA[j][k] += les * cA[i][k];
        }
    }

    for(i=N-1; i>=0; --i)
    {
        x[i]= - cA[i][N]/cA[i][i];
        for (j=0;j<i;++j)
        {
            cA[j][N] += x[i] * cA[j][i];
            cA[j][i]=0;
        }
    }

    cout << "Size: " << N << ", time: " << (MPI_Wtime()-t1)*1000 << "ms"<< endl;
    output_matrix()
}
