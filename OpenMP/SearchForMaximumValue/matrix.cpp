#include <iostream>
#include <math.h> 
#include <omp.h>
#include <stdio.h>
using namespace std;

int main()
{
	int n, m;
	cout << "Please, enter n: ";
	cin >> n;
	cout << endl << "now enter m: ";
	cin >> m;
	cout << endl;

	int** A = new int* [n];

	for (int i = 0; i < n; ++i) {
		A[i] = new int[m];
		for (int j = 0; j < m; ++j) {
			cout << "Please enter element:";
			cin >> A[i][j];
			cout << endl;
		}
	}

    int mins[n];
    int threads = omp_get_num_threads();
    #pragma omp parallel for schedule(auto)
    {for (int i = 0; i < n; ++i) mins[i] = A[i][0]; }

    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (mins[i] > A[i][j]) {
                mins[i] = A[i][j];
            }

        }
    }
    int minMax = mins[0];

    for (int i = 0; i < n; ++i) {
        if (minMax < mins[i]) {
            minMax = mins[i];
        }
    }
    cout << "Max of mins: " << minMax;
    return 0;


}

