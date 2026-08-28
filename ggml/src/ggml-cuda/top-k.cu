#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

static __global__ void k_top_k_merge_f32_i32(const float * src, const int * chunk_idx, int * cursors, int * dst,
                                             const int ncols, const int nrows, const int k, const int k_eff,
                                             const int chunk_size, const int n_chunks) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // zero-fill output slots beyond k_eff (when k > ncols)
    for (int i = k_eff + tid; i < k; i += blockDim.x) {
        dst[(size_t) row * k + i] = 0;
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }

    const float * src_row    = src + (size_t) row * ncols;
    int *        cursors_row = cursors + (size_t) row * n_chunks;
    int *        dst_row     = dst + (size_t) row * k;

    // sequential k-way merge of the per-chunk sorted (desc) index lists
    for (int i = 0; i < k_eff; ++i) {
        float best_val = -INFINITY;
        int   best_c   = 0;
        int   best_idx = 0;
        for (int c = 0; c < n_chunks; ++c) {
            const int chunk_ncols = (ncols - c * chunk_size < chunk_size) ? ncols - c * chunk_size : chunk_size;
            const int pos         = cursors_row[c];
            if (pos >= chunk_ncols) {
                continue;
            }
            // bitonic argsort writes each chunk row-major with stride chunk_ncols
            const int   idx = chunk_idx[(size_t) c * chunk_size * nrows + (size_t) row * chunk_ncols + pos];
            const float v   = src_row[c * chunk_size + idx];
            if (v > best_val) {
                best_val = v;
                best_c   = c;
                best_idx = idx;
            }
        }
        dst_row[i] = best_c * chunk_size + best_idx;
        ++cursors_row[best_c];
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    if (ncols <= 1024) {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    } else {
        // bitonic argsort uses ncols_pad threads per block, which exceeds the 1024 thread limit for ncols > 1024
        // split columns into <= 1024 chunks, argsort each chunk, then merge the per-chunk top-k candidates
        const int chunk_size = 1024;
        const int n_chunks   = (int) ((ncols + chunk_size - 1) / chunk_size);
        const int k_eff      = (int) std::min(k, ncols);

        ggml_cuda_pool_alloc<int>   chunk_idx_alloc(pool, (size_t) n_chunks * chunk_size * nrows);
        ggml_cuda_pool_alloc<float> src_chunk_alloc(pool, (size_t) chunk_size * nrows);
        ggml_cuda_pool_alloc<int>   cursors_alloc(pool, (size_t) n_chunks * nrows);
        int *   chunk_idx = chunk_idx_alloc.get();
        float * src_chunk = src_chunk_alloc.get();
        int *   cursors   = cursors_alloc.get();

        CUDA_CHECK(cudaMemsetAsync(cursors, 0, (size_t) n_chunks * nrows * sizeof(int), stream));

        for (int c = 0; c < n_chunks; ++c) {
            const int64_t col0        = (int64_t) c * chunk_size;
            const int     chunk_ncols = (int) std::min((int64_t) chunk_size, ncols - col0);
            CUDA_CHECK(cudaMemcpy2DAsync(src_chunk, chunk_ncols * sizeof(float), src0_d + col0, src0->nb[1],
                                         chunk_ncols * sizeof(float), nrows, cudaMemcpyDeviceToDevice, stream));
            argsort_f32_i32_cuda_bitonic(src_chunk, chunk_idx + (size_t) c * chunk_size * nrows, chunk_ncols, nrows,
                                         GGML_SORT_ORDER_DESC, stream);
        }

        const dim3 block_dims(256, 1, 1);
        const dim3 block_nums((unsigned) nrows, 1, 1);
        k_top_k_merge_f32_i32<<<block_nums, block_dims, 0, stream>>>(src0_d, chunk_idx, cursors, dst_d, (int) ncols, (int) nrows,
                                                                     (int) k, k_eff, chunk_size, n_chunks);
    }
#endif
}
