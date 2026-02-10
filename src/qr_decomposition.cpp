#include "qr_decomposition.hpp"
#include <iostream>
#include <cmath>

using namespace torch::indexing;


// Helper Functions
namespace {
    std::tuple<torch::Tensor, double> house(torch::Tensor x) {
        int64_t m = x.size(0);
        double sigma = (m > 1) ? torch::dot(x.index({Slice(1, m)}), x.index({Slice(1, m)})).item<double>() : 0.0;

        torch::Tensor v = torch::zeros_like(x);
        if (m > 1) {
            v.index_put_({Slice(1, m)}, x.index({Slice(1, m)}));
        }
        v[0] = 1.0;

        double beta = 0.0;
        double x1 = x[0].item<double>();

        if (sigma == 0) {
            beta = (x1 < 0) ? 2.0 : 0.0;
        } else {
            double mu = std::sqrt(x1 * x1 + sigma);
            if (x1 <= 0) {
                v[0] = x1 - mu;
            } else {
                v[0] = -sigma / (x1 + mu);
            }

            double v1 = v[0].item<double>();
            beta = 2 * (v1 * v1) / (sigma + v1 * v1);
            v = v / v1;
        }

        return std::make_tuple(v, beta);
    }

    std::tuple<double, double> givens(double a, double b) {
        double c, s;
        if (b == 0) {
            c = 1.0; s = 0.0;
        } else {
            if (std::abs(b) > std::abs(a)) {
                double tau = -a / b;
                s = 1.0 / std::sqrt(1 + tau * tau);
                c = s * tau;
            } else {
                double tau = -b / a;
                c = 1.0 / std::sqrt(1 + tau * tau);
                s = c * tau;
            }
        }
        return std::make_tuple(c, s);
    }
}


// QR Implementations

// Classical Gram-Schmidt
// Uses standard Gram-Schmidt orthogonalization
torch::Tensor qr_cgs(const torch::Tensor& A_in) {
    torch::Tensor A = A_in.clone();
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    torch::Tensor Q = torch::zeros({m, n}, A.options());
    torch::Tensor R = torch::zeros({n, n}, A.options());

    // Initialize R(0,0) with the norm of the first column
    double r00 = torch::norm(A.index({Slice(), 0})).item<double>();
    R.index_put_({0, 0}, r00);

    // Normalize the first column to get the first column of Q
    if (r00 != 0) {
        Q.index_put_({Slice(), 0}, A.index({Slice(), 0}) / r00);
    }

    // Iterate over the remaining columns
    for (int64_t k = 1; k < n; ++k) {
        // Compute projection coefficients (R entries)
        torch::Tensor q_prev = Q.index({Slice(), Slice(0, k)});
        torch::Tensor a_k = A.index({Slice(), k});
        torch::Tensor r_col = torch::matmul(q_prev.t(), a_k);
        R.index_put_({Slice(0, k), k}, r_col);

        // Orthogonalize current column against previous Q columns
        torch::Tensor z = a_k - torch::matmul(q_prev, r_col);

        // Compute the diagonal element of R
        double r_kk = torch::norm(z).item<double>();
        R.index_put_({k, k}, r_kk);

        // Normalize to get the next column of Q
        if (r_kk != 0) {
            Q.index_put_({Slice(), k}, z / r_kk);
        }
    }

    return R;
}

// Modified Gram-Schmidt
// Improves numerical stability over Classical Gram-Schmidt
torch::Tensor qr_mgs(const torch::Tensor& A_in) {
    torch::Tensor V = A_in.clone();
    int64_t m = V.size(0);
    int64_t n = V.size(1);

    torch::Tensor Q = torch::zeros({m, n}, V.options());
    torch::Tensor R = torch::zeros({n, n}, V.options());

    // Iterate through each column
    for (int64_t k = 0; k < n; ++k) {
        // Compute norm of current column (diagonal of R)
        double r_kk = torch::norm(V.index({Slice(), k})).item<double>();
        R.index_put_({k, k}, r_kk);

        // Normalize current column to form Q column
        if (r_kk != 0) {
            Q.index_put_({Slice(), k}, V.index({Slice(), k}) / r_kk);
        }

        // Update remaining columns
        for (int64_t j = k + 1; j < n; ++j) {
            // Compute projection of Q_k onto remaining columns
            torch::Tensor q_k = Q.index({Slice(), k});
            torch::Tensor v_j = V.index({Slice(), j});
            double r_kj = torch::dot(q_k, v_j).item<double>();
            R.index_put_({k, j}, r_kj);

            // Subtract projection from remaining columns (orthogonalize immediately)
            V.index_put_({Slice(), j}, v_j - q_k * r_kj);
        }
    }

    return R;
}

// In-Place MGS
// Overwrites A with Q. Returns R.
torch::Tensor qr_mgs_inplace(torch::Tensor& A) {
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    torch::Tensor R = torch::zeros({n, n}, A.options());

    for (int64_t k = 0; k < n; ++k) {
        torch::Tensor q_k = A.index({Slice(), k});
        double r_kk = torch::norm(q_k).item<double>();
        R.index_put_({k, k}, r_kk);

        if (r_kk != 0) {
            q_k.div_(r_kk);
        }

        for (int64_t j = k + 1; j < n; ++j) {
            torch::Tensor v_j = A.index({Slice(), j});

            double r_kj = torch::dot(q_k, v_j).item<double>();
            R.index_put_({k, j}, r_kj);

            v_j.add_(q_k, -r_kj);
        }
    }

    return R;
}

// Householder Reflections (Explicit)
torch::Tensor qr_householder_explicit(const torch::Tensor& A_in) {
    torch::Tensor A = A_in.clone();
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    std::vector<torch::Tensor> vs;
    std::vector<double> betas;

    int64_t steps = std::min(m, n);

    // Iterate through columns to compute Householder vectors
    for (int64_t j = 0; j < steps; ++j) {
        // Safe slicing now guaranteed
        torch::Tensor sub_col = A.index({Slice(j, m), j});
        auto result = house(sub_col);
        torch::Tensor v = std::get<0>(result);
        double beta = std::get<1>(result);

        vs.push_back(v);
        betas.push_back(beta);

        torch::Tensor sub_matrix = A.index({Slice(j, m), Slice(j, n)});
        torch::Tensor v_view = v.view({-1, 1});
        torch::Tensor v_dot_A = torch::matmul(v_view.t(), sub_matrix);

        torch::Tensor update = beta * torch::matmul(v_view, v_dot_A);
        A.index_put_({Slice(j, m), Slice(j, n)}, sub_matrix - update);
    }

    return torch::triu(A).index({Slice(0, steps), Slice()}).contiguous();
}

// Implicit Householder
torch::Tensor qr_householder_implicit(torch::Tensor& A) {
    int64_t m = A.size(0);
    int64_t n = A.size(1);
    int64_t steps = std::min(m, n);

    for (int64_t j = 0; j < steps; ++j) {
        torch::Tensor sub_col = A.index({Slice(j, m), j});
        auto result = house(sub_col);
        torch::Tensor v = std::get<0>(result);
        double beta = std::get<1>(result);

        if (beta != 0.0) {
            torch::Tensor sub_matrix = A.index({Slice(j, m), Slice(j, n)});

            torch::Tensor v_view = v.view({-1, 1});

            torch::Tensor w = torch::matmul(v_view.t(), sub_matrix);

            sub_matrix.addmm_(v_view, w, /*beta=*/1.0, /*alpha=*/-beta);
        }

        if (j < m - 1) {
            A.index_put_({Slice(j + 1, m), j}, v.index({Slice(1, m - j)}));
        }
    }

    return torch::triu(A).index({Slice(0, steps), Slice()}).contiguous();
}

// Givens Rotations (Explicit)
torch::Tensor qr_givens_explicit(const torch::Tensor& A_in) {
    torch::Tensor A = A_in.clone();
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    torch::Tensor Q = torch::eye(m, A.options());

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = m - 1; i > j; --i) {
            double a_top = A.index({i - 1, j}).item<double>();
            double a_bot = A.index({i, j}).item<double>();

            auto params = givens(a_top, a_bot);
            double c = std::get<0>(params);
            double s = std::get<1>(params);

            torch::Tensor row_top = A.index({i - 1, Slice()});
            torch::Tensor row_bot = A.index({i, Slice()});

            torch::Tensor new_top = c * row_top - s * row_bot;
            torch::Tensor new_bot = s * row_top + c * row_bot;

            A.index_put_({i - 1, Slice()}, new_top);
            A.index_put_({i, Slice()}, new_bot);

            torch::Tensor col_left = Q.index({Slice(), i - 1});
            torch::Tensor col_right = Q.index({Slice(), i});
            torch::Tensor new_left = c * col_left - s * col_right;
            torch::Tensor new_right = s * col_left + c * col_right;
            Q.index_put_({Slice(), i - 1}, new_left);
            Q.index_put_({Slice(), i}, new_right);
        }
    }

    return A.index({Slice(0, std::min(m, n)), Slice()}).contiguous();
}

// In-Place Givens
torch::Tensor qr_givens_inplace(torch::Tensor& A) {
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = m - 1; i > j; --i) {
            double a_top = A[i - 1][j].item<double>();
            double a_bot = A[i][j].item<double>();

            if (a_bot == 0.0) continue;

            auto params = givens(a_top, a_bot);
            double c = std::get<0>(params);
            double s = std::get<1>(params);

            auto row_top = A.index({i - 1, Slice(j, n)});
            auto row_bot = A.index({i, Slice(j, n)});

            torch::Tensor top_copy = row_top.clone();
            torch::Tensor bot_copy = row_bot.clone();

            row_top.copy_(top_copy).mul_(c).add_(bot_copy, -s);

            row_bot.copy_(top_copy).mul_(s).add_(bot_copy, c);

            A.index_put_({i, j}, 0.0);
        }
    }

    return A.index({Slice(0, std::min(m, n)), Slice()}).contiguous();
}
