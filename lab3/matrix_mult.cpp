#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <mpi.h>

using namespace std;

void read_matrix(const string& filename, vector<double>& matrix, int n) {
    ifstream file(filename, ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(matrix.data()), (size_t)n * n * sizeof(double));
    } else {
        cerr << "Error while opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void write_matrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename, ios::binary);
    if (file) {
        file.write(reinterpret_cast<const char*>(matrix.data()), (size_t)n * n * sizeof(double));
    }
}

void multiply_matrices_parallel(const vector<double>& A_local, const vector<double>& B, vector<double>& C_local, int rows_per_proc, int n) {
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int k = 0; k < n; ++k) {
            double temp = A_local[i * n + k];
            for (int j = 0; j < n; ++j) {
                C_local[i * n + j] += temp * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 5) {
        if (rank == 0) cerr << "Usage: mpiexec -n <p> ./exe <n> <A.bin> <B.bin> <C.bin>" << endl;
        MPI_Finalize();
        return 1;
    }

    int n = stoi(argv[1]);
    int rows_per_proc = n / world_size;

    vector<double> B(n * n);
    vector<double> A_local(rows_per_proc * n);
    vector<double> C_local(rows_per_proc * n, 0.0);

    if (rank == 0) {
        vector<double> A_full(n * n);
        read_matrix(argv[2], A_full, n);
        read_matrix(argv[3], B, n);

        MPI_Scatter(A_full.data(), rows_per_proc * n, MPI_DOUBLE, 
                    A_local.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, 0, MPI_DOUBLE, 
                    A_local.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    auto start = chrono::high_resolution_clock::now();

    multiply_matrices_parallel(A_local, B, C_local, rows_per_proc, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    vector<double> C_full;
    if (rank == 0) C_full.resize(n * n);

    MPI_Gather(C_local.data(), rows_per_proc * n, MPI_DOUBLE, 
               C_full.data(), rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << chrono::duration<double>(end - start).count(); // Без лишнего текста
    }

    MPI_Finalize();
    return 0;
}