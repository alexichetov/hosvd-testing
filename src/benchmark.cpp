#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>
#include <sys/resource.h>

#include "st_hosvd.hpp"
#include "qr_st_hosvd.hpp"

long get_peak_memory_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void run_benchmark(int64_t size, int64_t rank, const std::string& algo_type) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    // Generate Data
    std::vector<int64_t> shape = {size, size, size};
    torch::Tensor X = torch::randn(shape, options);
    std::vector<int64_t> ranks = {rank, rank, rank};

    // Force synchronization to ensure allocation is done
    torch::Tensor dummy = torch::zeros({1});

    // Measure Baseline Memory (Data only)
    long mem_data_loaded = get_peak_memory_kb();

    // Run Algorithm
    auto start = std::chrono::high_resolution_clock::now();

    if (algo_type == "ST") {
        st_hosvd(X, ranks);
    } else if (algo_type == "QR") {
        qr_st_hosvd(X, ranks);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Measure Peak Memory
    long mem_final_peak = get_peak_memory_kb();
    double peak_mb = mem_final_peak / 1024.0;

    // Calculate approximate overhead (Algorithm Peak - Data Load Peak)
    long mem_algo_overhead = mem_final_peak - mem_data_loaded;
    double overhead_mb = mem_algo_overhead / 1024.0;



    std::cout << size << "," << algo_type << ","
              << std::fixed << std::setprecision(6) << elapsed.count() << ","
              << std::fixed << std::setprecision(2) << peak_mb << ","
              << overhead_mb << std::endl;
}

int main(int argc, char* argv[]) {
    torch::NoGradGuard no_grad;

    if (argc < 2) {
        std::cerr << "Usage : ./benchmark ST/QR" << std::endl;
        return 1;
    }

    std::string algo = argv[1];

    std::cout << "Size,Algorithm,Time(s),Peak Memory(MB),Algo Overhead(MB)" << std::endl;

    std::vector<int64_t> sizes = {64, 128, 256, 512, 1024};
    int64_t fixed_rank = 10;

    for (int64_t s : sizes) {
        run_benchmark(s, fixed_rank, algo);
    }

    return 0;
}
