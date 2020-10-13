__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1; 
}

__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
    return a > a1 ? a + log1p(exp(a1 - a)) : a1 + log1p(exp(a - a1));
}

__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}

extern "C" __global__ void fwd_bwd_logspace(
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta_T,
    FLOAT* __restrict__ beta_stay, FLOAT* __restrict__ beta_move, 
    const FLOAT* __restrict__ stay_scores, const FLOAT* __restrict__ move_scores,
    int T, int N, int L
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= L) return;
    extern __shared__ FLOAT smem[];
    if (blockIdx.y == 0) {
        FLOAT a = ZERO, a1 = ZERO;
        a = alpha[bx * L + tx];
        for (int t = 0; t < T; t++) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = a; __syncthreads(); 
            if (tx > 0) {a1 = MUL(move_scores[(t * N + bx) * (L - 1) + tx - 1], buf[tx - 1]);}
            a = SUM(MUL(stay_scores[(t * N + bx) * L + tx], a), a1);
            alpha[((t + 1) * N + bx) * L + tx] = a;
        }
    }
    else {
        FLOAT b = ZERO, b1 = ZERO;
        b = beta_T[bx * L + tx];
        for (int t = T; t > 0; t--) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = b; __syncthreads();
            if (tx < L - 1) {
                b1 = MUL(buf[tx + 1], move_scores[(((t - 1) * N + bx) * (L - 1)) + tx]);
                beta_move[((t - 1) * N + bx) * L + tx] = b1;
            }
            b = MUL(b, stay_scores[(((t - 1) * N + bx) * L) + tx]);
            beta_stay[((t - 1) * N + bx) * L + tx] = b;
            b = SUM(b, b1);
        }
    }
  }

extern "C" __global__ void fwd_bwd_logspace_loop(
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta,
    FLOAT* __restrict__ beta_stay, FLOAT* __restrict__ beta_move, 
    const FLOAT* __restrict__ stay_scores, const FLOAT* __restrict__ move_scores,
    int T, int N, int L
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (blockIdx.y == 0) {
        FLOAT a;
        for (int t = 0; t < T; t++) {
            for (int j = tx; j < L; j += blockDim.x) {
                a = (j > 0) ? MUL(move_scores[(t * N + bx) * (L - 1) + j - 1], alpha[(t * N + bx) * L + j - 1]) : ZERO;
                alpha[((t + 1) * N + bx) * L + j] = SUM(MUL(stay_scores[(t * N + bx) * L + j], alpha[(t * N + bx) * L + j]), a);
            }
            __syncthreads();
        }
    }
    else {
        FLOAT b, b1;
        for (int t = T; t > 0; t--) {
            for (int j = L - blockDim.x + tx; j >= 0; j -= blockDim.x) {
                b1 = ZERO;
                if (j < L - 1) {
                    b1 = MUL(beta[(t * N + bx) * L + j + 1], move_scores[(((t - 1) * N + bx) * (L - 1)) + j]);
                    beta_move[((t - 1) * N + bx) * L + j] = b1;
                }
                b = MUL(beta[(t * N + bx) * L + j], stay_scores[(((t - 1) * N + bx) * L) + j]);
                beta_stay[((t - 1) * N + bx) * L + j] = b;
                beta[((t - 1) * N + bx) * L + j] = SUM(b, b1);
            }
            __syncthreads();
        }
    }
  }