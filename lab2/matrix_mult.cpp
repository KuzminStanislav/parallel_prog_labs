#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

using namespace std;

void read_matrix(const string& filename, vector<double>& matrix, int n) {
    ifstream file(filename, ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(matrix.data()), (size_t)n * n * sizeof(double));
    } else {
        cerr << "Error while opening file: " << filename << endl;
        exit(1);
    }
}

void write_matrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename, ios::binary);
    if (file) {
        file.write(reinterpret_cast<const char*>(matrix.data()), (size_t)n * n * sizeof(double));
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: <N> <file_A> <file_B> <file_C> <threads>" << endl;
        return 1;
    }

    int n = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_c = argv[4];
    int threads = stoi(argv[5]);

    omp_set_num_threads(threads);

    vector<double> A((size_t)n * n);
    vector<double> B((size_t)n * n);
    vector<double> C((size_t)n * n, 0.0);

    read_matrix(file_a, A, n);
    read_matrix(file_b, B, n);

    vector<double> BT((size_t)n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            BT[(size_t)i * n + j] = B[(size_t)j * n + i];
        }
    }

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            size_t row_offset = (size_t)i * n;
            size_t col_offset = (size_t)j * n;
            for (int k = 0; k < n; k++) {
                sum += A[row_offset + k] * BT[col_offset + k];
            }
            C[row_offset + j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << diff.count() << endl;

    write_matrix(file_c, C, n);

    return 0;
}