// parallel.cpp
// SIT315 - M2.T1P - Parallel matrix multiplication using std::thread
// Work is divided by rows - each thread handles its own chunk of rows

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// change N to test different matrix sizes
const int N = 1500;

// change this to test with 2, 4, 8, 12 threads etc
const int NUM_THREADS = 12;

double A[N][N], B[N][N], C[N][N];

void fillMatrix(double mat[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = (double)(rand() % 100) / 10.0;
}

// this is the function each thread runs
// every thread only works on the rows it was assigned
// startRow is where it begins, endRow is where it stops
void computeRows(int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                // standard dot product - row i of A times column j of B
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
    srand(42); // same seed as sequential so outputs will match

    // step 1 - fill matrices before threads start
    // threads will only READ from A and B so this is safe
    fillMatrix(A);
    fillMatrix(B);

    // step 2 - figure out how many rows each thread gets
    int rowsPerThread = N / NUM_THREADS;

    // step 3 - create a list to store all our thread objects
    vector<thread> threads;

    // step 4 - start the timer before launching threads
    auto startTime = high_resolution_clock::now();

    // step 5 - create and launch all threads
    for (int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;

        // last thread takes any leftover rows
        // handles cases where N doesn't divide evenly by NUM_THREADS
        int endRow = (t == NUM_THREADS - 1) ? N : startRow + rowsPerThread;

        // launch the thread and pass it its row range
        threads.push_back(thread(computeRows, startRow, endRow));
    }

    // step 6 - wait for every thread to finish before moving on
    // join() blocks the main thread until each worker thread completes
    for (auto& th : threads) {
        th.join();
    }

    // step 7 - stop timer only after ALL threads are done
    auto endTime = high_resolution_clock::now();
    double elapsed = duration<double, milli>(endTime - startTime).count();

    cout << "Matrix size: " << N << " x " << N << "\n";
    cout << "Number of threads: " << NUM_THREADS << "\n";
    cout << "Parallel execution time: " << elapsed << " ms\n";

    // save result - not part of timing
    saveToFile("parallel_output.txt");

    return 0;
}