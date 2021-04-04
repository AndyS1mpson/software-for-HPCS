#include <iostream>
#include <fstream>
#include <string>
#include<vector>
#include<omp.h>
using namespace std;

#pragma omp declare reduction (merge: vector<size_t>: omp_out.insert(omp_out.end(),omp_in.begin(),omp_in.end()))
int main() {
    vector<size_t> vec = {};
    string str, temp;
    ifstream in;
    in.open("in.txt");

    if (in.is_open()) {
        getline(in, str);
    }
    cout << "Please,enter a template: ";
    cin >> temp;

    bool is_template_found = false;
    int countChar = 0;
    int count = 0;
    double runtime;
    int i = 1;
    do {
        is_template_found = false;
        vec = {};
        countChar = 0;
        count = 0;
        runtime = omp_get_wtime();
        #pragma omp parallel for reduction(merge: vec) num_threads(i)
        for (size_t pos = 0; pos < str.size(); ++pos) {
            if (str.find(temp, pos) == pos) {
                is_template_found = true;
                vec.push_back(pos);
                #pragma omp atomic
                count = count + 1;
            }
        }

        cout << omp_get_num_threads() << "  threads" << endl;
        cout << count << "  number of results" << endl;
        for (int i = 0; i < count; ++i) {
            cout << vec[i] << endl;
        }
        runtime = omp_get_wtime() - runtime;
        cout << i << " threads, runtime is " << runtime << endl;

    } while (i < omp_get_thread_num());

    if (!is_template_found) cout << " string ' " << temp << " ' not found";


}