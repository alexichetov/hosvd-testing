#include "st_hosvd.hpp"
#include "utils.hpp"
#include <ATen/ops/linalg_eigh.h>

std::pair<torch::Tensor, std::vector<torch::Tensor>> st_hosvd(
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

        // Unfold tensor on mode-n
        torch::Tensor Yn = utils::unfold(core, n);

        // Compute Gram matrix S = Yn * Yn^T
        torch::Tensor S = torch::matmul(Yn, Yn.t());

        // Compute top Rn eigenvectors of S
        auto eig_result = at::linalg_eigh(S, "L");
        torch::Tensor evecs = std::get<1>(eig_result);

        int64_t dim = S.size(0);
        int64_t start_idx = (dim > rank) ? (dim - rank) : 0;

        torch::Tensor U_ascending = evecs.slice(/*dim=*/1, /*start=*/start_idx, /*end=*/dim);
        torch::Tensor U_n = U_ascending.flip({1});
        factors[n] = U_n;

        // Update X to be X x_n (U^(n))^T
        core = utils::mode_product(core, U_n.t(), n);
    }

    return {core, factors};
}
