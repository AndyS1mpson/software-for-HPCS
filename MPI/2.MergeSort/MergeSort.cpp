#include <iostream>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <mpi.h>

void merge(int* arr, int left, int m, int right) {
    int i, j, k;
    int n1 = m - left + 1;
    int n2 = right - m;
    int* lArr = new int[n1], * rArr = new int[n2];
    for (i = 0; i < n1; i++) lArr[i] = arr[left + i];
    for (j = 0; j < n2; j++) rArr[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = left;
    
    while (i < n1 || j < n2) {
        if (j >= n2 || (i < n1 && lArr[i] <= rArr[j])) {
            arr[k] = lArr[i];
            i++;
        }
        else {
            arr[k] = rArr[j];
            j++;
        }
        k++;
    }

    delete[] lArr;
    delete[] rArr;
}


void merge_sort(int* arr, int left, int right) {
    if (left < right) {
        int m = left + (right - left) / 2;
        merge_sort(arr, left, m);
        merge_sort(arr, m + 1, right);
        merge(arr, left, m, right);
    }
}

void parallel_merge_sort(int* arr, int l, int r, int thread_num, int max_threads, int thread_offset) {
    if (l >= r) {
        return;
    }

    int m = l + (r - l) / 2;
    int next_thread_num = thread_num + thread_offset;

    // if we have more free threads then...
    if (next_thread_num < max_threads) {
        MPI_Request msg_request;
        MPI_Status msg_status;

        MPI_Isend(arr + m + 1, r - m, MPI_INT, next_thread_num, thread_offset * 2, MPI_COMM_WORLD, &msg_request);
        parallel_merge_sort(arr, l, m, thread_num, max_threads, thread_offset * 2);
        MPI_Recv(arr + m + 1, r - m, MPI_INT, next_thread_num, thread_offset * 2, MPI_COMM_WORLD, &msg_status);
        merge(arr, l, m, r);

        // Just prevents a lot of warning in output, not necessary
        MPI_Wait(&msg_request, &msg_status);
    }
    else {
        parallel_merge_sort(arr, l, m, thread_num, max_threads, thread_offset * 2);
        parallel_merge_sort(arr, m + 1, r, thread_num, max_threads, thread_offset * 2);
        merge(arr, l, m, r);
    }
}



bool is_array_sorted(int* arr, int size) {
    for (int i = 0; i < size - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;

    return true;
}

int* array_generator(int size, int l = 500) {

    int* arr = new int[size];

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % l;
    }

    return arr;
}

void test(int n, int iterations, int l) {
    clock_t cpuTime = 0;
    double rTime = 0;
    for (int i = 0; i < iterations; i++) {
        int* arr = array_generator(n, l);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();
        merge_sort(arr, 0, n - 1);
        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        if (!is_array_sorted(arr, n)) {
            std::cout << "Array size error" << n << std::endl;
        }
        delete[] arr;

        cpuTime += (c_end - c_start);
        rTime += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
    std::cout << "Result for array with: size=" << n << ", iterations=" << iterations << std::endl;
    std::cout << "CPU: " << 1000.0 * cpuTime / (iterations * CLOCKS_PER_SEC) << " ms" << std::endl;
    std::cout << "Real time: " << rTime / iterations << " ms" << std::endl;
}

void parallel_test(int n, int iterations, int l) {
    int max_threads, thread_num;
    MPI_Comm_size(MPI_COMM_WORLD, &max_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_num);

    clock_t cpuTime = 0;
    double rTime = 0;

    for (int i = 0; i < iterations; i++) {
        int* arr = array_generator(n, l);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        parallel_merge_sort(arr, 0, n - 1, thread_num, max_threads, 1);

        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        if (!is_array_sorted(arr, n)) {
            std::cout << "Array size error" << n << std::endl;
        }
        delete[] arr;

        cpuTime += (c_end - c_start);
        rTime += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    std::cout << "Result for array with: size=" << n << ", iterations=" << iterations << std::endl;
    std::cout << "CPU: " << 1000.0 * cpuTime / iterations / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Real time: " << rTime / iterations << " ms" << std::endl;
}

void mpi_run(int argc, char** argv){
    int max_threads, thread_num;
    MPI_Status msg_status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_num);

    if (thread_num == 0) {
        return;
    }
    else {
        while (true) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msg_status);

            int p = msg_status.MPI_TAG;
            int source = msg_status.MPI_SOURCE;
            if (p == 0) {
                int dummy;
                MPI_Recv(&dummy, 0, MPI_INT, source, p, MPI_COMM_WORLD, &msg_status);
                MPI_Finalize();
                exit(0);
            }
            else {
                int* arr;
                int arr_size;

                MPI_Get_count(&msg_status, MPI_INT, &arr_size);
                arr = new int[arr_size];
                MPI_Recv(arr, arr_size, MPI_INT, source, p, MPI_COMM_WORLD, &msg_status);

                parallel_merge_sort(arr, 0, arr_size - 1, thread_num, max_threads, p);

                MPI_Send(arr, arr_size, MPI_INT, source, p, MPI_COMM_WORLD);
                delete[] arr;
            }
        }
    }
}


int main(int argc, char** argv) {
    std::cout << "Sequential version of merge sort" << std::endl;
    test(10000, 70, 100);
    test(20000, 70, 200);
    test(40000, 70, 300);
    test(60000, 70, 400);
    test(80000, 70, 500);


    mpi_run(argc, argv);
    int max_threads;
    MPI_Comm_size(MPI_COMM_WORLD, &max_threads);
    
    std::cout << "Parallel version of merge sort" << std::endl;
    parallel_test(10000, 70, 100);
    parallel_test(20000, 70, 200);
    parallel_test(40000, 70, 300);
    parallel_test(60000, 70, 400);
    parallel_test(80000, 70, 500);

    for (int i = 1; i < max_threads; i++) {
        MPI_Send(0, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}