#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

void read_matrix(const string& filename, vector<double>& matrix, int n){
    ifstream file(filename, ios::binary);
    if (file){
        file.read(reinterpret_cast<char*>(matrix.data()), n * n * sizeof(double));
    } else {
        cerr << "Error while openning file: " << filename << endl;
        exit(1);
    }
}

void write_matrix(const string& filename, const vector<double>& matrix, int n){
    ofstream file(filename, ios::binary);
    if (file){
        file.write(reinterpret_cast<const char*>(matrix.data()), n * n * sizeof(double));
    }
}

int main(int argc, char* argv[]){
    if (argc != 5){
        cerr << "Incorrect args!" << endl;
        return 1;
    }

    int n = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_c = argv[4];

    vector<double> A(n * n);
    vector<double> B(n * n);
    vector<double> C(n * n, 0.0);

    read_matrix(file_a, A, n);
    read_matrix(file_b, B, n);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            double temp = A[i * n + k];
            for(int j = 0; j < n; ++j){
                C[i * n + j] += temp * B[k * n + j];
            }
        }
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    write_matrix(file_c, C, n);
    cout << elapsed.count() << endl;
    return 0;
}