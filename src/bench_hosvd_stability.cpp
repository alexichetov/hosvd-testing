#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <functional>
#include <string>
#include <cstdlib>
#include <algorithm>
#include "qr_decomposition.hpp"
#include "utils.hpp"

using namespace torch::indexing;

// Helper to generate a tensor with a specific multilinear condition number
torch::Tensor generate_controlled_tensor(int64_t size, double condition_number, const torch::TensorOptions& options) {
    int64_t core_dim = 10;
    torch::Tensor core = torch::zeros({core_dim, core_dim, core_dim}, options);

    // Create logarithmic decay from 1.0 down to (1/cond)
    auto s_log = torch::linspace(0.0, std::log10(1.0 / condition_number), core_dim, options);
    auto s_vals = torch::pow(10.0, s_log);

    for (int i = 0; i < core_dim; ++i) {
        core[i][i][i] = s_vals[i];
    }

    // Create random orthogonal factor matrices
    std::vector<torch::Tensor> factors;
    for (int i = 0; i < 3; ++i) {
        auto [Q, R] = torch::linalg_qr(torch::randn({size, core_dim}, options));
        factors.push_back(Q);
    }

    // Compose tensor
    torch::Tensor X = core.clone();
    for (int i = 0; i < 3; ++i) {
        X = utils::mode_product(X, factors[i], i);
    }
    return X;
}

// Define the signature for a solver that returns the top 'r' left singular vectors
typedef std::function<torch::Tensor(const torch::Tensor&, int64_t)> SvdSolver;

// ST-HOSVD implementation that accepts a custom matrix decomposition strategy
double test_hosvd_accuracy(const torch::Tensor& X, int64_t rank, SvdSolver solver) {
    torch::Tensor core = X.clone();
    int64_t ndim = X.dim();
    std::vector<torch::Tensor> factors(ndim);

    // Run ST-HOSVD
    for (int64_t n = 0; n < ndim; ++n) {
        torch::Tensor Yn = utils::unfold(core, n);
        torch::Tensor Un = solver(Yn, rank);
        factors[n] = Un;
        core = utils::mode_product(core, Un.t(), n);
    }

    // Reconstruct
    torch::Tensor X_rec = core.clone();
    for (int i = 0; i < ndim; ++i) {
        X_rec = utils::mode_product(X_rec, factors[i], i);
    }

    return (X - X_rec).norm().item<double>() / X.norm().item<double>();
}

int main(int argc, char* argv[]) {
    int64_t seed = 42;
    auto dtype = torch::kFloat64;
    std::string precision_name = "Double";

    int64_t tensor_size = 32;
    int64_t target_rank = 5; //compressing from 10

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed") {
            if (i + 1 < argc) seed = std::stoll(argv[++i]);
            else { std::cerr << "Error: --seed needs arg.\n"; return 1; }
        } else if (arg == "--size") {
            if (i + 1 < argc) tensor_size = std::stoll(argv[++i]);
            else { std::cerr << "Error: --size needs arg.\n"; return 1; }
        } else if (arg == "--rank") {
            if (i + 1 < argc) target_rank = std::stoll(argv[++i]);
            else { std::cerr << "Error: --rank needs arg.\n"; return 1; }
        } else if (arg == "float") {
            dtype = torch::kFloat32;
            precision_name = "Float";
        } else if (arg == "double") {
            dtype = torch::kFloat64;
            precision_name = "Double";
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    std::cerr << "Config -> Seed: " << seed
              << " | Size: " << tensor_size
              << " | Rank: " << target_rank
              << " | Precision: " << precision_name << std::endl;

    torch::manual_seed(seed);
    torch::NoGradGuard no_grad;
    auto options = torch::TensorOptions().dtype(dtype);

    std::cout << "ConditionNumber,Gram,CGS,MGS,H_Explicit,H_Implicit,Givens,MGS_Inplace,Givens_Inplace,LibTorch_QR" << std::endl;

    for (double exp = 0.0; exp <= 16.0; exp += 1.0) {
        double cond = std::pow(10.0, exp);
        torch::Tensor X = generate_controlled_tensor(tensor_size, cond, options);

        auto solve_gram = [](const torch::Tensor& Yn, int64_t r) {
            auto S = torch::matmul(Yn, Yn.t());
            auto [evals, evecs] = torch::linalg_eigh(S);
            return evecs.slice(1, evecs.size(0) - r, evecs.size(0)).flip({1});
        };

        auto solve_qr_cgs = [](const torch::Tensor& Yn, int64_t r) {
            auto R = qr_cgs(Yn.t());
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_mgs = [](const torch::Tensor& Yn, int64_t r) {
            auto R = qr_mgs(Yn.t());
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_hexp = [](const torch::Tensor& Yn, int64_t r) {
            auto R = qr_householder_explicit(Yn.t());
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        // do not know why this is failing so hard
        auto solve_qr_himp = [](const torch::Tensor& Yn, int64_t r) {
            auto A = Yn.t().clone();
            auto R = qr_householder_implicit(A);
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_givens = [](const torch::Tensor& Yn, int64_t r) {
            auto R = qr_givens_explicit(Yn.t());
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_mgs_inplace = [](const torch::Tensor& Yn, int64_t r) {
            auto A = Yn.t().clone();
            auto R = qr_mgs_inplace(A);
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_givens_inplace = [](const torch::Tensor& Yn, int64_t r) {
            auto A = Yn.t().clone();
            auto R = qr_givens_inplace(A);
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto solve_qr_libtorch = [](const torch::Tensor& Yn, int64_t r) {
            auto result = torch::linalg_qr(Yn.t());
            torch::Tensor R = std::get<1>(result);
            auto [U, S, V] = torch::linalg_svd(R.t(), false);
            return U.slice(1, 0, r);
        };

        auto safe_test = [&](SvdSolver solver) {
            try {
                return test_hosvd_accuracy(X, target_rank, solver);
            } catch (...) {
                return 1.0;
            }
        };

        std::cout << std::scientific << std::setprecision(4) << cond << ","
                  << safe_test(solve_gram) << ","
                  << safe_test(solve_qr_cgs) << ","
                  << safe_test(solve_qr_mgs) << ","
                  << safe_test(solve_qr_hexp) << ","
                  << safe_test(solve_qr_himp) << ","
                  << safe_test(solve_qr_givens) << ","
                  << safe_test(solve_qr_mgs_inplace) << ","
                  << safe_test(solve_qr_givens_inplace) << ","
                  << safe_test(solve_qr_libtorch) << std::endl;
    }

    return 0;
}
