#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <utility>


double oddEvenSorting(int* arr, int n)
{
    bool is_sorted = false;
    int init = 0;
    double time1 = omp_get_wtime();
    while (!is_sorted)
    {
        is_sorted = true;
        for (int i = init; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                std::swap(arr[i], arr[i + 1]);
                is_sorted = false;
            }
        }
        init = 1 - init;
    }
    double time2 = omp_get_wtime();
    return (time2 - time1) * 1000.0;
}

double oddEvenSortingOMP(int* arr, int n)
{
    bool is_sorted = false;
    int init = 0;
    double time1 = omp_get_wtime();
    while (!is_sorted)
    {
        is_sorted = true;
        #pragma omp parallel for
        for (int i = init; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                std::swap(arr[i], arr[i + 1]);
                is_sorted = false;
            }
        }
        init = 1 - init;
    }
    double time2 = omp_get_wtime();
    return (time2 - time1) * 1000.0;
}

int* arr_generator(int n)
{
    static int c;
    c++;
    int* res = new int[n];
    srand(c);
    for (int i = 0; i < n; ++i)
    {
        res[i] = rand() % 1000;
    }
    return res;
}

void test(int n, int c)
{
    double t = 0.0;
    for (int i = 0; i < c; ++i)
    {
        int* arr = arr_generator(n);
        t += oddEvenSorting(arr, n);
        delete[] arr;
    }
    printf("size: %d, time: %.2lf ms\n", n, t / c);
}

void test_omp(int n, int cnt)
{
    double t = 0.0;
    for (int i = 0; i < cnt; ++i)
    {
        int* arr = arr_generator(n);
        t += oddEvenSortingOMP(arr, n);
        delete[] arr;
    }
    printf("size: %d, time with omp: %.2lf ms\n", n, t / cnt);
}


int main()
{
    int num_threads = 4;
    
    omp_set_num_threads(num_threads);
    test(100, 10);
    test_omp(100, 10);
    test(500, 10);
    test_omp(500, 10);
    test(1000, 10);
    test_omp(1000, 10);
    test(1300, 10);
    test_omp(1300, 10);
    test(2000, 10);
    test_omp(2000, 10);

    return 0;
}