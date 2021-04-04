#include <omp.h>
#include <iostream>

using namespace std;

int main() {
    int rowsA, colsA, rowsB, colsB;
    double** A, ** B, ** C;

    cout << "Input rows and columns number: ";
    cin >> rowsA;
    cin >> colsA;
    cout << "Input rows and columns number:";
    cin >> rowsB;
    cin >> colsB;

    if (colsA != rowsB) {
        cout << "Composition is impossible!";
        cin.get(); cin.get();
        return 0;
    }

    A = new double* [rowsA];
    cout << "Enter elements for first matrix" << endl;
    for (int i = 0; i < rowsA; ++i) {
        A[i] = new double[colsA];
        for (int j = 0; j < colsA; ++j) {
            cout << "A[" << i << "][" << j << "]=";
            cin >> A[i][j];
        }
    }

    B = new double* [rowsB];
    cout << "Enter elements for second matrix" << endl;
    for (int i = 0; i < rowsB; ++i) {
        B[i] = new double[colsB];
        for (int j = 0; j < colsB; ++j) {
            cout << "B[" << i << "][" << j << "]=";
            cin >> B[i][j];
        }
    }

    C = new double* [rowsA];
    for (int i = 0; i < rowsA; ++i) {
        C[i] = new double[colsB];
    }

    #pragma omp parallel for collapse(2) schedule(static) shared(A,B)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
    #pragma omp critical
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    cout << "Matrix of composition:" << endl;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }
    cin.get(); cin.get();
    return 0;

}