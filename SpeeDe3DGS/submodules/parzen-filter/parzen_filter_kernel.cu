/*
 * CUDA kernel for the Parzen-window 3D filter.
 *
 * parzen_density_kernel<D_COMP> (2D joint density)
 *   Computes the full joint density[N, D, T] over a (depth × time) grid.
 *   One CUDA block per Gaussian; one thread per t-grid point.
 *   fd[K] and ts[K] for each Gaussian are loaded into shared memory once,
 *   then each thread accumulates D density values for its t-grid point.
 *   No [N,D,T,K] intermediate; D accumulators live in registers per thread.
 *
 * Template parameter D_COMP: compile-time D so inner loops fully unroll.
 * Explicit instantiations for D ∈ {5, 10, 20}.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ── helper: log-space Weibull kernel ────────────────────────────────────────
// Weibull PDF: (k/lam) * (u/lam)^(k-1) * exp(-(u/lam)^k),  u >= 0
// Computed in log-space to avoid float overflow for large u or large k.
__device__ __forceinline__ float weibull_log_safe(float u, float wk, float wlam)
{
    if (u <= 0.0f) return -1e38f;           // mask: kernel = 0 for u <= 0
    float u_norm    = u / wlam;
    float log_u     = logf(u_norm);
    float pow_k_log = wk * log_u;           // log(u_norm^k)
    if (pow_k_log > 80.0f) return -1e38f;  // u_norm^k overflows → kernel ≈ 0
    float log_kern = logf(wk / wlam) + (wk - 1.0f) * log_u - expf(pow_k_log);
    return log_kern;
}

// ── 2D joint density [N, D, T] ──────────────────────────────────────────────
//
// Grid:   (N, 1, 1)  — one block per Gaussian
// Block:  (T_round, 1, 1)  — one thread per t-grid point (padded to warp)
// Shmem:  2 * K * sizeof(float)  — fd[K] and ts[K] loaded cooperatively
//
// Each thread accumulates D density values for its (n, t) pair.
// The D_COMP density accumulators and x_grid values live in registers.
// Output layout: density[n, d, t] = density[n * D * T + d * T + t]

template <int D_COMP>
__global__ void parzen_density_kernel(
    const float* __restrict__ fd_data,   // [N, K]
    const float* __restrict__ ts_data,   // [N, K]
    const float* __restrict__ x_grid,    // [D_COMP]  shared across Gaussians
    const float* __restrict__ t_grid,    // [T]       shared across Gaussians
    float*       __restrict__ density,   // [N, D_COMP, T]
    int   N,
    int   K,
    int   T,
    float h_x_inv,
    float h_t_inv,
    float wk,
    float wlam,
    float norm_factor)                   // 1 / (2 * h_x * h_t)
{
    int n = blockIdx.x;
    int t = threadIdx.x;
    // Gate ONLY on n here. Threads with t >= T must NOT return yet: the block
    // size (blockDim.x = T padded up to a warp) can be < K, so every thread is
    // needed for the cooperative shared-memory load below to cover all K slots,
    // and every thread must reach the __syncthreads() barrier. The t >= T guard
    // is applied *after* the barrier (see below).
    if (n >= N) return;

    // ── Load fd[K] and ts[K] for this Gaussian into shared memory ────────────
    extern __shared__ float shmem[];    // 2 * K floats
    float* sfd = shmem;                 // [K]
    float* sts = shmem + K;             // [K]

    // All blockDim.x threads participate, stride blockDim.x → covers k = 0..K-1
    // with no gaps even when K > blockDim.x.
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sfd[k] = fd_data[n * K + k];
        sts[k] = ts_data[n * K + k];
    }
    __syncthreads();

    // Now safe to drop the padding threads: shared memory is fully populated and
    // every thread has crossed the barrier.
    if (t >= T) return;

    // ── Load shared x_grid into registers ────────────────────────────────────
    float xg[D_COMP];
    #pragma unroll
    for (int d = 0; d < D_COMP; d++) xg[d] = x_grid[d];

    // ── D accumulators for this (n, t) thread ────────────────────────────────
    float acc[D_COMP];
    #pragma unroll
    for (int d = 0; d < D_COMP; d++) acc[d] = 0.0f;

    const float INV_SQRT_2PI = 0.3989422804f;
    float tg = t_grid[t];

    // ── Single pass over K observations ──────────────────────────────────────
    for (int k = 0; k < K; k++) {
        float ts_k = sts[k];
        if (ts_k < 0.0f) continue;         // sentinel: empty slot
        float fd_k = sfd[k];

        float t_diff = (tg - ts_k) * h_t_inv;
        float t_w    = INV_SQRT_2PI * expf(-0.5f * t_diff * t_diff);

        #pragma unroll
        for (int d = 0; d < D_COMP; d++) {
            float u  = (xg[d] - fd_k) * h_x_inv;
            float lk = weibull_log_safe(u, wk, wlam);
            float xk = (lk < -80.0f) ? 0.0f : expf(lk);
            acc[d]  += xk * t_w;
        }
    }

    // ── Write density[n, 0..D-1, t] to global memory ─────────────────────────
    // Layout: density[n * D_COMP * T + d * T + t]
    int base = n * D_COMP * T;
    #pragma unroll
    for (int d = 0; d < D_COMP; d++)
        density[base + d * T + t] = acc[d] * norm_factor;
}

// ── Explicit instantiations ──────────────────────────────────────────────────
template __global__ void parzen_density_kernel<5>(
    const float*, const float*, const float*, const float*, float*,
    int, int, int, float, float, float, float, float);
template __global__ void parzen_density_kernel<10>(
    const float*, const float*, const float*, const float*, float*,
    int, int, int, float, float, float, float, float);
template __global__ void parzen_density_kernel<20>(
    const float*, const float*, const float*, const float*, float*,
    int, int, int, float, float, float, float, float);

// ── Host-side launcher ───────────────────────────────────────────────────────
torch::Tensor parzen_density_cuda(
    torch::Tensor fd_data,    // [N, K]  float32 CUDA
    torch::Tensor ts_data,    // [N, K]  float32 CUDA
    torch::Tensor x_grid,     // [D]     float32 CUDA  (1D, shared)
    torch::Tensor t_grid,     // [T]     float32 CUDA  (1D, shared)
    float h_x,
    float h_t,
    float wk,
    float wlam)
{
    TORCH_CHECK(fd_data.is_cuda(),   "fd_data must be a CUDA tensor");
    TORCH_CHECK(fd_data.dtype() == torch::kFloat32, "fd_data must be float32");

    int N = fd_data.size(0);
    int K = fd_data.size(1);
    int D = x_grid.size(0);
    int T = t_grid.size(0);

    TORCH_CHECK(T <= 1024, "parzen_density: T must be <= 1024 (got ", T, ")");

    auto density = torch::zeros({N, D, T}, fd_data.options());

    // Round block size up to next warp multiple so the SM is kept busy
    int threads = ((T + 31) / 32) * 32;
    int blocks  = N;
    int shmem   = 2 * K * sizeof(float);

    float h_x_inv     = 1.0f / h_x;
    float h_t_inv     = 1.0f / h_t;
    float norm_factor = 1.0f / (2.0f * h_x * h_t);

    auto fd_ptr  = fd_data.contiguous().data_ptr<float>();
    auto ts_ptr  = ts_data.contiguous().data_ptr<float>();
    auto xg_ptr  = x_grid.contiguous().data_ptr<float>();
    auto tg_ptr  = t_grid.contiguous().data_ptr<float>();
    auto out_ptr = density.data_ptr<float>();

    if (D == 5)
        parzen_density_kernel<5><<<blocks, threads, shmem>>>(
            fd_ptr, ts_ptr, xg_ptr, tg_ptr, out_ptr,
            N, K, T, h_x_inv, h_t_inv, wk, wlam, norm_factor);
    else if (D == 10)
        parzen_density_kernel<10><<<blocks, threads, shmem>>>(
            fd_ptr, ts_ptr, xg_ptr, tg_ptr, out_ptr,
            N, K, T, h_x_inv, h_t_inv, wk, wlam, norm_factor);
    else if (D == 20)
        parzen_density_kernel<20><<<blocks, threads, shmem>>>(
            fd_ptr, ts_ptr, xg_ptr, tg_ptr, out_ptr,
            N, K, T, h_x_inv, h_t_inv, wk, wlam, norm_factor);
    else
        TORCH_CHECK(false, "parzen_density: D must be 5, 10, or 20 (got ", D, ")");

    return density;  // [N, D, T]
}
