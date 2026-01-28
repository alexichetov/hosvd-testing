#pragma once

#include <torch/torch.h>
#include <vector>
#include <numeric>
#include <iostream>

namespace utils {

    // Unfold tensor into a matrix along a specific mode (0-indexed)
    inline torch::Tensor unfold(const torch::Tensor& tensor, int64_t mode) {
        int64_t ndim = tensor.dim();
        if (mode >= ndim || mode < 0) {
            throw std::runtime_error("Invalid mode for unfolding");
        }

        std::vector<int64_t> perm;
        perm.reserve(ndim);
        perm.push_back(mode);
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != mode) {
                perm.push_back(i);
            }
        }

        torch::Tensor permuted = tensor.permute(perm);
        int64_t first_dim = tensor.size(mode);
        return permuted.reshape({first_dim, -1});
    }

    // Fold matrix back into a tensor (inverse of unfold)
    inline torch::Tensor fold(const torch::Tensor& matrix, int64_t mode, c10::IntArrayRef shape) {
        int64_t ndim = shape.size();

        std::vector<int64_t> perm_shape;
        perm_shape.reserve(ndim);
        perm_shape.push_back(shape[mode]);
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != mode) perm_shape.push_back(shape[i]);
        }

        torch::Tensor reshaped = matrix.reshape(perm_shape);

        std::vector<int64_t> inv_perm(ndim);
        int64_t counter = 1;
        inv_perm[mode] = 0;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != mode) {
                inv_perm[i] = counter++;
            }
        }

        return reshaped.permute(inv_perm);
    }

    // Mode-n Product: Tensor x_n Matrix
    inline torch::Tensor mode_product(const torch::Tensor& tensor, const torch::Tensor& matrix, int64_t mode) {
        torch::Tensor unfolded = unfold(tensor, mode);

        torch::Tensor res_mat = torch::matmul(matrix, unfolded);

        std::vector<int64_t> new_shape = tensor.sizes().vec();
        new_shape[mode] = matrix.size(0);

        return fold(res_mat, mode, new_shape);
    }
}
