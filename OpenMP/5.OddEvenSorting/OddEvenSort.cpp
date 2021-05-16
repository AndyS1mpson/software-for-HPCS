#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


class Arr {
public:
    Arr() {
        int* arr = nullptr;
        int size;
    }

    void generate_arr(int size, int range = 1e+9, int seed = 42) {
        std::mt19937 gen;
        gen.seed(seed);

        this->size = size;
        this->arr = new int[this->size];

        for (int i = 0; i < this->size; i++) {
            this->arr[i] = gen() * range;
        }

        return;
    }

    void sort() {
        bool is_sorted;
        int start_value = 0;

        while (!is_sorted || start_value != 0) {
            is_sorted = true;

            for (int i = start_value; i < this->size - 1; i += 2)
                if (this->arr[i] > this->arr[i + 1]) {
                    std::swap(this->arr[i], this->arr[i + 1]);
                    is_sorted = false;
                }

            start_value = 1 - start_value;
        }
    }

    int& operator[](int i) const {
        if (i < 0 || i >= this->size) throw std::out_of_range{ "Vector::operator[]" };
        return this->arr[i];
    }

    int get_size() {
        return this->size;
    }

    ~Arr() {
        if (this->arr != nullptr)
            delete[] this->arr;
        this->arr = nullptr;
    }

private:
    int* arr;
    int size;
};


bool is_arr_sorted(Arr& arr) {
    for (int i = 0; i < arr.get_size() - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;

    return true;
}


void test(int n, int iters) {
    clock_t t_cpu = 0;
    double t = 0;

    for (int i = 0; i < iters; i++) {
        Arr arr;
        arr.generate_arr(n, 1e+4, i);
        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();
        arr.sort();
        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();
        if (!is_arr_sorted(arr)) {
            std::cout << "Array size error" << n << std::endl;
        }
        t_cpu += (c_end - c_start);
        t += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    std::cout << "Arr of size " << n << " with mean by " << iters << " iterations" << std::endl;
    std::cout << "CPU time: " << 1000.0 * t_cpu / iters / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Time has passed: " << t / iters << " ms" << std::endl;
}


int main() {
    std::cout << std::endl;
    std::cout << "Threads number: " << 1 << std::endl;
    test(1000, 10);
    test(2000, 10);
    test(4000, 10);
    test(8000, 10);
    test(12000, 10);
    return 0;
}