__device__ __forceinline__ FLOAT max3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = a > a1 ? a : a1; 
    return maxa > a2 ? maxa : a2;
}

__device__ __forceinline__ FLOAT logsumexp3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = max3(a, a1, a2); 
    return maxa + log(exp(a-maxa) + exp(a1-maxa) + exp(a2-maxa));
}

__device__ __forceinline__ FLOAT sum3(FLOAT a, FLOAT a1, FLOAT a2) {return a + a1 + a2;}
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}

extern "C" __global__ void fwd_bwd_banded(
    FLOAT* __restrict__ alpha_T, 
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta,  
    const FLOAT* __restrict__ scores,  const bool* __restrict__ repeat_mask, 
    const long* __restrict__ lengths, const long* __restrict__ window_starts,
    int N, int L, int W
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= W) return;
    extern __shared__ FLOAT smem[];
    int T = (int) lengths[bx];
    if (blockIdx.y == 0) {
        int prev = window_starts[bx];
        FLOAT a, a1, a2;
        a = alpha[bx * W + tx];
        for (int t = 0; t < T; t++) {
            int pos = (int) window_starts[(t + 1) * N + bx];
            int ix = tx + pos - prev;
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = a; __syncthreads();
            a = (ix < W) ? buf[ix] : ZERO;
            a1 = ((ix > 0) && (ix - 1 < W)) ? buf[ix - 1] : ZERO;
            a2 = ((ix > 1) && (!repeat_mask[bx * L + pos + tx])) ? buf[ix - 2] : ZERO; 
            a = MUL(scores[(t * N + bx) * W + tx], SUM(a, a1, a2));
            alpha[((t + 1) * N + bx) * W + tx] = a;
            prev = pos;
        }
        alpha_T[bx * W + tx] = a;
    }
    else {
        int prev = (int) window_starts[T * N + bx];
        FLOAT b, b1, b2;
        b = beta[(T * N + bx) * W + tx];
        for (int t = T; t > 0; t--) {
            int pos = (int) window_starts[(t - 1) * N + bx];
            int ix = tx + pos - prev;
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = MUL(b, scores[(((t - 1) * N + bx) * W) + tx]);
            __syncthreads();
            b = (ix >= 0) ? buf[ix] : ZERO;
            b1 = ((ix + 1 >= 0) && (ix + 1 < W)) ? buf[ix + 1] : ZERO;
            b2 = ((ix + 2 < W) && (!repeat_mask[bx * L + pos + tx + 2])) ? buf[ix + 2] : ZERO;
            b = beta[((t - 1) * N + bx) * W + tx] = SUM(b, b1, b2);
            prev = pos;
        }
    }
  }