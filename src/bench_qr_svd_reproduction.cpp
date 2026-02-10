#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <functional>
#include "qr_decomposition.hpp"

// Generates the 80x80 matrix with singular values from 10^0 to 10^-18
// We generate in Double first to ensure identical matrices, then cast if needed.
torch::Tensor generate_matrix(int64_t size, torch::ScalarType dtype) {
    torch::manual_seed(42);
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    auto log_s = torch::linspace(0, -18, size, options);
    auto S_diag = torch::diag(torch::pow(10.0, log_s));

    auto U = std::get<0>(torch::linalg_qr(torch::randn({size, size}, options)));
    auto V = std::get<0>(torch::linalg_qr(torch::randn({size, size}, options)));

    torch::Tensor A_double = torch::matmul(U, torch::matmul(S_diag, V.t()));
    return A_double.to(dtype);
}

torch::Tensor get_singular_values_from_r(const torch::Tensor& R) {
    auto [U, S, V] = torch::linalg_svd(R, /*full_matrices=*/false);
    return S;
}

void run_test_for_precision(const std::string& prec_name, torch::ScalarType dtype) {
    int64_t size = 80;
    auto A = generate_matrix(size, dtype);

    std::map<std::string, std::function<torch::Tensor(const torch::Tensor&)>> algos;

    algos["CGS"]             = [](const torch::Tensor& X) { return qr_cgs(X); };
    algos["MGS"]             = [](const torch::Tensor& X) { return qr_mgs(X); };
    algos["MGS_Inplace"]     = [](const torch::Tensor& X) { auto X_c = X.clone(); return qr_mgs_inplace(X_c); };
    algos["Householder_Exp"] = [](const torch::Tensor& X) { return qr_householder_explicit(X); };
    algos["Householder_Imp"] = [](const torch::Tensor& X) { auto X_c = X.clone(); return qr_householder_implicit(X_c); };
    algos["Givens_Exp"]      = [](const torch::Tensor& X) { return qr_givens_explicit(X); };
    algos["Givens_Imp"]      = [](const torch::Tensor& X) { auto X_c = X.clone(); return qr_givens_inplace(X_c); };
    algos["LibTorch"]        = [](const torch::Tensor& X) { return std::get<1>(torch::linalg_qr(X, "r")); };

    for (auto const& [name, func] : algos) {
        try {
            torch::Tensor R = func(A);
            torch::Tensor S = get_singular_values_from_r(R);

            auto S_cpu = S.to(torch::kCPU).to(torch::kFloat64);
            auto acc = S_cpu.accessor<double, 1>();

            for(int64_t i=0; i<size; ++i) {
                std::cout << i << "," << name << "," << prec_name << "," << acc[i] << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error running " << name << " (" << prec_name << "): " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "Index,Algorithm,Precision,Value" << std::endl;

    run_test_for_precision("Float", torch::kFloat32);

    run_test_for_precision("Double", torch::kFloat64);

    torch::manual_seed(42);
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto log_s = torch::linspace(0, -18, 80, options);
    auto S_true = torch::pow(10.0, log_s);
    auto acc_true = S_true.accessor<double, 1>();

    for(int64_t i=0; i<80; ++i) {
        std::cout << i << ",True,Double," << acc_true[i] << std::endl;
    }

    return 0;
}
