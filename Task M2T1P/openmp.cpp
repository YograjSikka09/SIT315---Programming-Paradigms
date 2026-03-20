// openmp.cpp
// SIT315 - M2.T1P - Matrix multiplication using OpenMP
// Only one pragma line needed to parallelise the entire multiplication loop

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// change N to test different matrix sizes
const int N = 1500;

// change this to test with 2, 4, 8, 12 threads
const int NUM_THREADS = 12;

double A[N][N], B[N][N], C[N][N];

void fillMatrix(double mat[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = (double)(rand() % 100) / 10.0;
}

// OpenMP parallelises this function automatically
// the pragma line tells the compiler to split the outer loop across threads
// each thread gets a chunk of rows to work on - same idea as std::thread
// but we did not write any of that splitting logic ourselves
void multiplyOMP() {
    omp_set_num_threads(NUM_THREADS);

    // this single line is all that is needed to parallelise the loop
    // schedule(static) means rows are divided equally among threads upfront
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void saveToFile(const string& filename) {
    ofstream out(filename);
    if (!out) {
        cerr << "Could not open file\n";
        return;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            out << C[i][j];
            if (j < N - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
    cout << "Result saved to " << filename << "\n";
}

int main() {
    srand(42); // same seed as before so output matches sequential

    // step 1 - fill matrices before parallel section starts
    fillMatrix(A);
    fillMatrix(B);

    // step 2 - start timer before calling the OpenMP function
    auto startTime = high_resolution_clock::now();

    // step 3 - run the parallel multiplication
    // OpenMP creates and manages all threads automatically here
    multiplyOMP();

    // step 4 - stop timer after multiplication is fully done
    auto endTime = high_resolution_clock::now();
    double elapsed = duration<double, milli>(endTime - startTime).count();

    cout << "Matrix size: " << N << " x " << N << "\n";
    cout << "Threads used (OpenMP): " << NUM_THREADS << "\n";
    cout << "OpenMP execution time: " << elapsed << " ms\n";

    // save result - outside timing as always
    saveToFile("openmp_output.txt");

    return 0;
}