#pragma once
#include <torch/torch.h>
#include <vector>

std::pair<torch::Tensor, std::vector<torch::Tensor>> qr_st_hosvd(
    const torch::Tensor& X,
    const std::vector<int64_t>& target_ranks
);
