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

extern "C" __global__ void fwd_bwd_logspace(
    FLOAT* __restrict__ alpha_T, 
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta,  
    const FLOAT* __restrict__ scores,  const bool* __restrict__ repeat_mask, 
    const long* __restrict__ lengths,
    int N, int L
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= L) return;
    extern __shared__ FLOAT smem[];
    int T = (int) lengths[bx];
    if (blockIdx.y == 0) {
        FLOAT a = ZERO, a1 = ZERO, a2 = ZERO;
        a = alpha[bx * L + tx];
        for (int t = 0; t < T; t++) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = a; __syncthreads(); 
            if (tx > 0) a1 = buf[tx - 1];
            if (tx > 1) if (!repeat_mask[bx * L + tx]) a2 = buf[tx - 2];
            a = MUL(scores[(t * N + bx) * L + tx], SUM(a, a1, a2));
            alpha[((t + 1) * N + bx) * L + tx] = a;
        }
        alpha_T[bx*L + tx] = a;
    }
    else {
        FLOAT b = ZERO, b1 = ZERO, b2 = ZERO;
        b = beta[(T * N + bx) * L + tx];
        for (int t = T; t > 0; t--) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            b = buf[tx] = MUL(b, scores[(((t - 1) * N + bx) * L) + tx]);
            __syncthreads(); 
            if (tx < L - 1) b1 = buf[tx+1];
            if (tx < L - 2) if (!repeat_mask[bx * L + tx + 2]) b2 = buf[tx + 2];
            b = beta[((t - 1) * N + bx) * L + tx] = SUM(b, b1, b2);
        }
    }
  }