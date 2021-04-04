#include <omp.h>
#include <iostream>
using namespace std;

int main() {
    int rows, cols, vecSize;
    double** A, * vec, * vecStr, * vecCols, * vecBlock;

    cout << "Enter amount of rows and columns for matrix:";
    cin >> rows;
    cin >> cols;
    cout << "Enter size of vector:";
    cin >> vecSize;

    if (cols != vecSize) {
        cout << "Composition is impossible";
        cin.get(); cin.get();
        return 0;
    }

    A = new double* [rows];
    cout << "Enter elements for matrix:" << endl;
    for (int i = 0; i < rows; ++i) {
        A[i] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            cout << "A[" << i << "][" << j << "]= ";
            cin >> A[i][j];
        }
    }

    vec = new double[vecSize];
    cout << "Enter elements for vector:" << endl;
    for (int i = 0; i < vecSize; ++i) {
        cout << "vec[" << i << "]=";
        cin >> vec[i];
    }

    vecStr = new double[vecSize];
    vecCols = new double[vecSize];
    vecBlock = new double[vecSize];

    double runtime;

    #pragma omp parallel for
    for (int i = 0; i < vecSize; ++i) {
        vecStr[i] = 0;
        vecBlock[i] = 0;
        vecCols[i] = 0;
    }
    int counter = 1;

    do {
        runtime = omp_get_wtime() * 1000;
        #pragma omp parallel for shared(A,vec,vecStr) num_threads(counter)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                #pragma omp atomic
                vecStr[i] += A[i][j] * vec[j];
            }
        }
        runtime = omp_get_wtime() * 1000 - runtime;
        cout << counter << " threads, lines time is " << runtime << endl;

        runtime = omp_get_wtime() * 1000;
        #pragma omp parallel for shared(A,vec,vecCols) num_threads(counter)
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < cols; ++i) {
                #pragma omp atomic
                vecCols[i] += A[i][j] * vec[j];
            }
        }
        runtime = omp_get_wtime() * 1000 - runtime;
        cout << counter << " threads, columns time is" << runtime << endl;

        runtime = omp_get_wtime() * 1000;
        int h = rows / counter;
        int w = cols / counter;
        #pragma omp parallel shared(A,vec,vecBlock) num_threads(counter)
        for (int blocks = 0; blocks < counter * counter; ++blocks) {
            int i = blocks / counter;
            int j = blocks % counter;
            for (int k = i * h; k < (i + 1) * h; ++k) {
                for (int l = j * w; l < (j + 1) * w; ++l) {
                    #pragma omp critical
                    vecBlock[k] = A[k][l] * vec[l];
                }
            }
        }
        runtime = omp_get_wtime() * 1000 - runtime;
        cout << counter << " threads, blocks time is " << runtime * 1000 << endl;
    } while (counter < omp_get_thread_num());

    cin.get(); cin.get();
    return 0;

}