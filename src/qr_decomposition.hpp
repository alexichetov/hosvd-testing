#pragma once
#include <torch/torch.h>

// Classical Gram-Schmidt
torch::Tensor qr_cgs(const torch::Tensor& A);

// Modified Gram-Schmidt
torch::Tensor qr_mgs(const torch::Tensor& A);

// In-Place MGS
// Overwrites A with Q.
torch::Tensor qr_mgs_inplace(torch::Tensor& A);

// Householder Reflections (Explicit)
torch::Tensor qr_householder_explicit(const torch::Tensor& A);

// Householder Reflections (Implicit)
// Overwrites A with the Householder vectors.
// Returns: R (extracted from the upper triangle of the modified A)
torch::Tensor qr_householder_implicit(torch::Tensor& A);

// Givens Rotations (Explicit)
torch::Tensor qr_givens_explicit(const torch::Tensor& A);

// In-Place Givens
// Overwrites A with R.
// Returns: R (which is the modified A)
torch::Tensor qr_givens_inplace(torch::Tensor& A);
