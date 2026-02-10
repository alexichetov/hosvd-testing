#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <iomanip>
#include <sys/resource.h>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "utils.hpp"
#include "qr_decomposition.hpp"

using namespace torch::indexing;

long get_peak_memory_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

// Generate a 5D tensor
torch::Tensor generate_5d_tensor(const std::vector<int64_t>& dims, double condition_number, torch::TensorOptions options) {

    int64_t min_dim = *std::min_element(dims.begin(), dims.end());

    torch::Tensor core = torch::zeros(dims, options);

    torch::Tensor diagonal_values = torch::logspace(0, std::log10(1.0/condition_number), min_dim, 10.0, torch::kFloat64);
    diagonal_values = diagonal_values.to(options.dtype());

    for(int64_t i = 0; i < min_dim; ++i) {
        core.index_put_({i, i, i, i, i}, diagonal_values[i]);
    }

    torch::Tensor X = core;
    for(size_t i = 0; i < dims.size(); ++i) {
        int64_t d = dims[i];
        auto qr = torch::linalg_qr(torch::randn({d, d}, options));
        torch::Tensor Q = std::get<0>(qr);
        X = utils::mode_product(X, Q, i);
    }

    return X;
}

int64_t determine_rank(const torch::Tensor& singular_values, double threshold_energy) {
    int64_t n = singular_values.size(0);
    double total_energy_sq = torch::sum(torch::pow(singular_values, 2)).item<double>();
    double current_energy_sq = 0.0;

    for(int64_t r = n; r > 0; --r) {
        double sigma = singular_values[r-1].item<double>();
        current_energy_sq += sigma * sigma;
        if (std::sqrt(current_energy_sq) > threshold_energy) {
            return r;
        }
    }
    return 1;
}


struct DecompResult {
    torch::Tensor core;
    std::vector<torch::Tensor> factors;
    double runtime_sec;
    long memory_kb;
};

typedef std::function<torch::Tensor(const torch::Tensor&)> QrFunc;

DecompResult run_st_hosvd_gram(const torch::Tensor& X, double epsilon) {
    long mem_baseline = get_peak_memory_kb();
    auto start = std::chrono::high_resolution_clock::now();

    torch::Tensor core = X.clone();
    int64_t ndim = X.dim();
    std::vector<torch::Tensor> factors(ndim);

    double norm_X = X.norm().item<double>();
    double threshold = epsilon * norm_X / std::sqrt((double)ndim);

    for (int64_t n = 0; n < ndim; ++n) {
        torch::Tensor Yn = utils::unfold(core, n);
        torch::Tensor S = torch::matmul(Yn, Yn.t());

        auto eig_res = torch::linalg_eigh(S);
        torch::Tensor evals = std::get<0>(eig_res);
        torch::Tensor evecs = std::get<1>(eig_res);

        int64_t dim_size = evals.size(0);
        double discarded_energy_sq = 0.0;
        int64_t cut_index = 0;

        for(int64_t i=0; i<dim_size; ++i) {
            double val = evals[i].item<double>();
            if(val < 0) val = 0;
            discarded_energy_sq += val;
            if(std::sqrt(discarded_energy_sq) > threshold) {
                cut_index = i;
                break;
            }
        }

        torch::Tensor Un = evecs.slice(1, cut_index, dim_size).flip({1});
        factors[n] = Un;
        core = utils::mode_product(core, Un.t(), n);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long mem_peak = get_peak_memory_kb();

    return {core, factors, std::chrono::duration<double>(end-start).count(), mem_peak - mem_baseline};
}

DecompResult run_st_hosvd_qr_optimized(const torch::Tensor& X, double epsilon, QrFunc qr_solver) {
    long mem_baseline = get_peak_memory_kb();
    auto start = std::chrono::high_resolution_clock::now();

    torch::Tensor core = X.clone();
    int64_t ndim = X.dim();
    std::vector<torch::Tensor> factors(ndim);

    double norm_X = X.norm().item<double>();
    double threshold = epsilon * norm_X / std::sqrt((double)ndim);

    for (int64_t n = 0; n < ndim; ++n) {
        torch::Tensor Yn = utils::unfold(core, n);
        torch::Tensor Y_t = Yn.t();
        torch::Tensor R = qr_solver(Y_t);
        torch::Tensor L = R.t();

        auto svd_res = torch::linalg_svd(L, /*full_matrices=*/false);
        torch::Tensor U = std::get<0>(svd_res);
        torch::Tensor S = std::get<1>(svd_res);

        int64_t rank = determine_rank(S, threshold);
        torch::Tensor Un = U.slice(1, 0, rank);
        factors[n] = Un;
        core = utils::mode_product(core, Un.t(), n);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long mem_peak = get_peak_memory_kb();

    return {core, factors, std::chrono::duration<double>(end-start).count(), mem_peak - mem_baseline};
}

int main(int argc, char* argv[]) {
    torch::manual_seed(42);

    if (argc < 9) {
        std::cerr << "Usage: ./bench_compression --algo <name> --cond <val> --eps <val> --prec <float|double>" << std::endl;
        return 1;
    }

    std::string algo = "";
    double cond = 1.0;
    double eps = 1e-4;
    std::string prec = "double";

    for(int i=1; i<argc; i+=2) {
        std::string key = argv[i];
        if (i+1 >= argc) break;
        std::string val = argv[i+1];
        if (key == "--algo") algo = val;
        else if (key == "--cond") cond = std::stod(val);
        else if (key == "--eps") eps = std::stod(val);
        else if (key == "--prec") prec = val;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    if (prec == "float") {
        options = torch::TensorOptions().dtype(torch::kFloat32);
    }

    std::vector<int64_t> dims = {64, 64, 16, 8, 8};

    torch::Tensor X = generate_5d_tensor(dims, cond, options);
    double original_size_elements = (double)X.numel();

    DecompResult res;

    try {
        if (algo == "Gram") {
            res = run_st_hosvd_gram(X, eps);
        } else if (algo == "LibTorch_QR") {
            res = run_st_hosvd_qr_optimized(X, eps, [](const torch::Tensor& T){
                return std::get<1>(torch::linalg_qr(T, "r"));
            });
        } else if (algo == "MGS_InPlace") {
            res = run_st_hosvd_qr_optimized(X, eps, [](const torch::Tensor& T){
                torch::Tensor T_copy = T.clone();
                return qr_mgs_inplace(T_copy);
            });
        } else if (algo == "Householder_Imp") {
            res = run_st_hosvd_qr_optimized(X, eps, [](const torch::Tensor& T){
                torch::Tensor T_copy = T.clone();
                return qr_householder_implicit(T_copy);
            });
        } else {
            std::cerr << "Unknown algorithm: " << algo << std::endl;
            return 1;
        }

        torch::Tensor X_rec = res.core.clone();
        for(size_t i=0; i<res.factors.size(); ++i) {
            X_rec = utils::mode_product(X_rec, res.factors[i], i);
        }

        double error = (X.to(torch::kFloat64) - X_rec.to(torch::kFloat64)).norm().item<double>() / X.to(torch::kFloat64).norm().item<double>();

        double core_elems = (double)res.core.numel();
        double factor_elems = 0;
        for(auto& f : res.factors) factor_elems += f.numel();
        double ratio = original_size_elements / (core_elems + factor_elems);

        std::cout << algo << ","
                  << prec << ","
                  << std::scientific << std::setprecision(2) << cond << ","
                  << std::scientific << std::setprecision(2) << eps << ","
                  << std::fixed << std::setprecision(6) << res.runtime_sec << ","
                  << res.memory_kb << ","
                  << std::scientific << std::setprecision(6) << error << ","
                  << std::fixed << std::setprecision(2) << ratio << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error running " << algo << ": " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
