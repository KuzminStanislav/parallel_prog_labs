#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void read_matrix(const string& filename, vector<double>& matrix, int n) {
    ifstream file(filename, ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(matrix.data()), n * n * sizeof(double));
    } else {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
}

void write_matrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename, ios::binary);
    if (file) {
        file.write(reinterpret_cast<const char*>(matrix.data()), n * n * sizeof(double));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: <n> <file_a> <file_b> <file_c> [block_size]" << endl;
        return 1;
    }

    int n = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_c = argv[4];
    int block_size = (argc == 6) ? stoi(argv[5]) : 16;

    size_t size = n * n * sizeof(double);
    vector<double> h_A(n * n), h_B(n * n), h_C(n * n);

    read_matrix(file_a, h_A, n);
    read_matrix(file_b, h_B, n);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(block_size, block_size);
    dim3 blocksPerGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    auto start = chrono::high_resolution_clock::now();

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    chrono::duration<double> elapsed = end - start;
    cout << elapsed.count() << endl;

    write_matrix(file_c, h_C, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}