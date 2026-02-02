#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "st_hosvd.hpp"
#include "qr_st_hosvd.hpp"
#include "utils.hpp"

torch::Tensor reconstruct(const torch::Tensor& core, const std::vector<torch::Tensor>& factors) {
    torch::Tensor recon = core.clone();
    for (size_t n = 0; n < factors.size(); ++n) {
        recon = utils::mode_product(recon, factors[n], n);
    }
    return recon;
}

void test_algorithm(
    const std::string& name,
    std::function<std::pair<torch::Tensor, std::vector<torch::Tensor>>(const torch::Tensor&, const std::vector<int64_t>&)> func,
    const torch::Tensor& X,
    const std::vector<int64_t>& ranks,
    double expected_max_error
) {
    std::cout << "Testing : " << name << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(X, ranks);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    torch::Tensor S = result.first;
    std::vector<torch::Tensor> Us = result.second;
    torch::Tensor X_rec = reconstruct(S, Us);

    double norm_X = X.norm().item<double>();
    double norm_diff = (X - X_rec).norm().item<double>();
    double rel_error = norm_diff / norm_X;

    std::cout << "Time : " << elapsed.count() << "s" << std::endl;
    std::cout << "Relative Error : " << rel_error << std::endl;

    if (rel_error < expected_max_error) {
        std::cout << "Result : SUCCESS" << std::endl;
    } else {
        std::cout << "Result : HIGH ERROR" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Default seed
    int64_t seed = 42;

    // Check for seed
    if (argc > 1) {
        try {
            seed = std::stoll(argv[1]);
            std::cout << "Using seed : " << seed << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Using default seed 42." << std::endl;
        }
    } else {
        std::cout << "No seed provided. Using default seed 42." << std::endl;
    }

    torch::manual_seed(seed);
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    int64_t size = 512;

    // Test 1: Synthetic Low-Rank
    std::cout << "Test 1 : Synthetic Low-Rank Tensor" << std::endl;
    torch::Tensor true_core = torch::randn({10, 10, 10}, options);
    std::vector<torch::Tensor> true_factors;
    for(int i=0; i<3; ++i) {
        auto qr = torch::linalg_qr(torch::randn({size, 10}, options), "reduced");
        true_factors.push_back(std::get<0>(qr));
    }

    torch::Tensor X_low = true_core.clone();
    for(int i=0; i<3; ++i) {
        X_low = utils::mode_product(X_low, true_factors[i], i);
    }

    std::vector<int64_t> target_ranks = {10, 10, 10};
    test_algorithm("ST-HOSVD", st_hosvd, X_low, target_ranks, 1e-10);
    test_algorithm("QR ST-HOSVD", qr_st_hosvd, X_low, target_ranks, 1e-10);

    // Test 2: Random Full-Rank
    std::cout << "Test 2 : Random Full-Rank Tensor" << std::endl;
    torch::Tensor X_rand = torch::randn({size, size, size}, options);
    test_algorithm("ST-HOSVD", st_hosvd, X_rand, target_ranks, 1.0);
    test_algorithm("QR ST-HOSVD", qr_st_hosvd, X_rand, target_ranks, 1.0);

    // Test 3: Even worse synthetic
    torch::Tensor core_diagonal = torch::zeros({10, 10, 10}, options);
    auto s_vals = torch::pow(10.0, torch::linspace(0, -12, 10, options));
    for (int i = 0; i < 10; ++i) {
      core_diagonal[i][i][i] = s_vals[i];
    }

    std::vector<torch::Tensor> factors;
    for (int i = 0; i < 3; ++i) {
      auto qr = torch::linalg_qr(torch::randn({size, 10}, options), "reduced");
      factors.push_back(std::get<0>(qr));
    }

    torch::Tensor x = core_diagonal.clone();
    for (int i = 0; i < 3; ++i) {
      x = utils::mode_product(x, factors[i], i);
    }

    test_algorithm("ST-HOSVD", st_hosvd, x, target_ranks, 1e-10);
    test_algorithm("QR ST-HOSVD", qr_st_hosvd, x, target_ranks, 1e-10);


    return 0;
}
