#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>

// Use ATen headers for specific linalg operations
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_cholesky.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/linalg_solve_triangular.h>

// Generates a random (M x N) matrix with a specific Condition Number.
torch::Tensor generate_ill_conditioned_matrix(int64_t rows, int64_t cols, double condition_number, const torch::TensorOptions& options) {
    auto U_qr = at::linalg_qr(torch::randn({rows, rows}, options), "reduced");
    auto U = std::get<0>(U_qr).slice(1, 0, cols);

    auto V_qr = at::linalg_qr(torch::randn({cols, cols}, options), "reduced");
    auto V = std::get<0>(V_qr);

    auto low = std::log10(1.0 / condition_number);
    auto high = std::log10(1.0);
    auto s_log = torch::linspace(high, low, cols, options);
    auto S_diag = torch::pow(10.0, s_log);

    auto S = torch::diag(S_diag);

    return torch::matmul(U, torch::matmul(S, V.t()));
}

struct Stats {
    double mean = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = 0.0;
};

Stats compute_stats(const std::vector<double>& errors) {
    Stats s;
    double sum = 0.0;
    for (double e : errors) {
        sum += e;
        if (e < s.min) s.min = e;
        if (e > s.max) s.max = e;
    }
    s.mean = sum / errors.size();
    return s;
}

int main() {
    torch::manual_seed(42);
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    int64_t M = 100;
    int64_t N = 10;
    int num_trials = 10; // Number of runs per condition number

    std::cout << "ConditionNumber,"
              << "GramAvg,GramMin,GramMax,"
              << "QRAvg,QRMin,QRMax" << std::endl;

    for (double exp = 0.0; exp <= 18.0; exp += 0.5) {
        double cond_num = std::pow(10.0, exp);

        std::vector<double> gram_errors;
        std::vector<double> qr_errors;

        for (int t = 0; t < num_trials; ++t) {
            // Setup
            torch::Tensor A = generate_ill_conditioned_matrix(M, N, cond_num, options);
            torch::Tensor x_true = torch::randn({N, 1}, options);
            torch::Tensor b = torch::matmul(A, x_true);
            double x_norm = x_true.norm().item<double>();

            // Gram
            try {
                torch::Tensor Gram = torch::matmul(A.t(), A);
                torch::Tensor Atb = torch::matmul(A.t(), b);
                torch::Tensor L = at::linalg_cholesky(Gram);
                torch::Tensor x_gram = at::cholesky_solve(Atb, L);

                double err = (x_true - x_gram).norm().item<double>() / x_norm;
                gram_errors.push_back(err);
            } catch (...) {
                gram_errors.push_back(1.0);
            }

            // QR
            try {
                auto qr_res = at::linalg_qr(A, "reduced");
                torch::Tensor Q = std::get<0>(qr_res);
                torch::Tensor R = std::get<1>(qr_res);
                torch::Tensor Qtb = torch::matmul(Q.t(), b);
                torch::Tensor x_qr = at::linalg_solve_triangular(R, Qtb, /*upper=*/true);

                double err = (x_true - x_qr).norm().item<double>() / x_norm;
                qr_errors.push_back(err);
            } catch (...) {
                qr_errors.push_back(1.0);
            }
        }

        Stats s_gram = compute_stats(gram_errors);
        Stats s_qr = compute_stats(qr_errors);

        std::cout << cond_num << ","
                  << s_gram.mean << "," << s_gram.min << "," << s_gram.max << ","
                  << s_qr.mean << "," << s_qr.min << "," << s_qr.max
                  << std::endl;
    }

    return 0;
}
