#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using Matrix = std::vector<double>;

inline double& at(Matrix& M, int n, int i, int j) {
    return M[static_cast<std::size_t>(i) * n + j];
}

inline const double& at(const Matrix& M, int n, int i, int j) {
    return M[static_cast<std::size_t>(i) * n + j];
}


void fill_random(Matrix& M, int n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& x : M) {
        x = dist(rng);
    }
}


void matmul_basic(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(B, n, k, j);
            }
            at(C, n, i, j) = sum;
        }
    }
}


void transpose(const Matrix& B, Matrix& BT, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            at(BT, n, j, i) = at(B, n, i, j);
        }
    }
}


void matmul_basic_transposed(const Matrix& A, const Matrix& BT, Matrix& C, int n) {
   
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(BT, n, j, k);
            }
            at(C, n, i, j) = sum;
        }
    }
}


void matmul_omp(const Matrix& A, const Matrix& B, Matrix& C, int n, int num_threads) {
   
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(B, n, k, j);
            }
            at(C, n, i, j) = sum;
        }
    }
}

template <typename Func>
double benchmark(Func f, int repeats = 3) {
    using clock = std::chrono::high_resolution_clock;
    double best_ms = std::numeric_limits<double>::max();

    for (int r = 0; r < repeats; ++r) {
        auto start = clock::now();
        f();
        auto end   = clock::now();
        double ms  = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < best_ms) best_ms = ms;
    }
    return best_ms;
}

int main() {
    
    std::vector<int> sizes = {256, 512, 1024};


    std::mt19937_64 rng(42);

    
    std::vector<int> thread_counts;
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif

   
    int candidates[] = {1, 2, 4, 8, 16, 32};
    for (int t : candidates) {
        if (t <= max_threads) {
            thread_counts.push_back(t);
        }
    }
    if (thread_counts.empty()) {
        thread_counts.push_back(1);
    }

 
    std::cout << "size,variant,threads,time_ms,speedup,efficiency\n";
    std::cout << std::fixed << std::setprecision(3);

    for (int n : sizes) {
        std::size_t total_elems = static_cast<std::size_t>(n) * n;

        Matrix A(total_elems);
        Matrix B(total_elems);
        Matrix BT(total_elems);
        Matrix C(total_elems);

        fill_random(A, n, rng);
        fill_random(B, n, rng);
        transpose(B, BT, n);

 
        double t_basic = benchmark([&]() {
            matmul_basic(A, B, C, n);
        }, 3);

        double speedup_basic    = 1.0;
        double efficiency_basic = 1.0;

        std::cout << n << ",basic,1,"
                  << t_basic << ","
                  << speedup_basic << ","
                  << efficiency_basic << "\n";

        double t_vec = benchmark([&]() {
            matmul_basic_transposed(A, BT, C, n);
        }, 3);

        double speedup_vec    = t_basic / t_vec;
        double efficiency_vec = speedup_vec; 

        std::cout << n << ",vectorized,1,"
                  << t_vec << ","
                  << speedup_vec << ","
                  << efficiency_vec << "\n";

    
        for (int threads : thread_counts) {
            double t_par = benchmark([&]() {
#ifdef _OPENMP
                matmul_omp(A, B, C, n, threads);
#else
            
                matmul_basic(A, B, C, n);
#endif
            }, 3);

            double speedup_par    = t_basic / t_par;
            double efficiency_par = speedup_par / threads;

            std::cout << n << ",parallel," << threads << ","
                      << t_par << ","
                      << speedup_par << ","
                      << efficiency_par << "\n";
        }
    }

    return 0;
}
