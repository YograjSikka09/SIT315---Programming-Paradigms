// sequential.cpp
// SIT315 - M2.T1P - Simple matrix multiplication

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// change N to test different sizes - 500, 1000, 1500 etc
const int N = 1500;

double A[N][N], B[N][N], C[N][N];

// fills a matrix with random numbers between 0.0 and 9.9
void fillMatrix(double mat[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = (double)(rand() % 100) / 10.0;
}

// the actual multiplication logic
// for each cell C[i][j], we take row i from A and column j from B
// multiply them element by element and sum everything up
void multiplyMatrices() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// writes the result matrix to a text file
// this is NOT included in the timing - just for saving output
void saveToFile(const string& filename) {
    ofstream out(filename);
    if (!out) {
        cerr << "Could not open file: " << filename << "\n";
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
    // fixed seed so we get the same random values every run
    // makes it easier to compare outputs across sequential and parallel versions
    srand(42);

    // step 1 - fill both matrices before we start timing anything
    fillMatrix(A);
    fillMatrix(B);

    // step 2 - start the clock just before multiplication begins
    auto startTime = high_resolution_clock::now();

    multiplyMatrices();

    // stop the clock right after multiplication finishes
    auto endTime = high_resolution_clock::now();

    // calculate how long it took in milliseconds
    double elapsed = duration<double, milli>(endTime - startTime).count();

    cout << "Matrix size: " << N << " x " << N << "\n";
    cout << "Sequential execution time: " << elapsed << " ms\n";

    // save result to file - outside timing window
    saveToFile("sequential_output.txt");

    return 0;
}
