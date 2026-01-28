#include "qr_st_hosvd.hpp"
#include "utils.hpp"
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_svd.h>

std::pair<torch::Tensor, std::vector<torch::Tensor>> qr_st_hosvd(
    const torch::Tensor& X,
    const std::vector<int64_t>& target_ranks
) {
    torch::Tensor core = X.clone();
    int64_t ndim = X.dim();
    std::vector<torch::Tensor> factors(ndim);

    if (target_ranks.size() != static_cast<size_t>(ndim)) {
        throw std::runtime_error("Target ranks size must match tensor dimensions.");
    }

    for (int64_t n = 0; n < ndim; ++n) {
        int64_t rank = target_ranks[n];

        // Unfold tensor on mode-n -> X_(n)
        torch::Tensor X_n = utils::unfold(core, n);

        // LQ Decompose X_(n) = L Q
        torch::Tensor Y_t = X_n.t();

        // "r" vs "reduced" ?
        auto qr_result = at::linalg_qr(Y_t, "r");     // to be replaced with better alg?
        torch::Tensor R_hat = std::get<1>(qr_result); // discarding unused Q, but still in memory?

        torch::Tensor L = R_hat.t();

        // Compute top SVD of L = U * S * V^T
        auto svd_result = at::linalg_svd(L, /*full_matrices=*/false);
        torch::Tensor U = std::get<0>(svd_result);

        // Truncate U to target rank
        torch::Tensor U_truncated = U.slice(/*dim=*/1, /*start=*/0, /*end=*/rank);

        factors[n] = U_truncated;

        // Update X to be X x_n (U^(n))^T
        core = utils::mode_product(core, U_truncated.t(), n);
    }

    return {core, factors};
}
