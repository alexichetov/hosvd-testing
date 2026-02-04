#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include "qr_decomposition.hpp"

double get_peak_memory_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Usage : ./bench_qr_memory <ALGO> <M> <N> <float|double>" << std::endl;
        return 1;
    }

    std::string algo = argv[1];
    int64_t M = std::stol(argv[2]);
    int64_t N = std::stol(argv[3]);
    std::string prec = argv[4];

    torch::NoGradGuard no_grad;

    auto options = torch::TensorOptions().dtype(
        (prec == "float") ? torch::kFloat32 : torch::kFloat64
    );

    try {
        // Allocation & Baseline Measurement
        // Force sync/allocation
        torch::Tensor A = torch::randn({M, N}, options);

        // Measure memory before algorithm
        double mem_baseline = get_peak_memory_mb();

        // Execution
        if (algo == "CGS") {
            auto R = qr_cgs(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "MGS") {
            auto R = qr_mgs(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "MGS_Inplace") {
            auto R = qr_mgs_inplace(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "Householder_Explicit") {
            auto R = qr_householder_explicit(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "Householder_Implicit") {
            auto R = qr_householder_implicit(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "Givens_Explicit") {
            auto R = qr_givens_explicit(A);
            volatile double d = R[0][0].item<double>();
        }
        else if (algo == "Givens_Inplace") {
            auto R = qr_givens_inplace(A);
            volatile double d = R[0][0].item<double>();
        }
        else {
            std::cerr << "Unknown algo : " << algo << std::endl;
            return 1;
        }

        // Peak Measurement
        double mem_peak = get_peak_memory_mb();

        std::cout << algo << ","
                  << prec << ","
                  << M << ","
                  << N << ","
                  << mem_peak << ","
                  << mem_baseline << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Torch Error : " << e.msg() << std::endl;
        return 1;
    }

    return 0;
}
