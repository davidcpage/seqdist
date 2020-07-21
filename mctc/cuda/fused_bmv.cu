__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1; 
}

__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
    FLOAT maxa = max2(a, a1); 
    return maxa + log(exp(a-maxa) + exp(a1-maxa));
}

__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}

extern "C" __global__ void fwd(
    FLOAT* __restrict__ alpha,
    const FLOAT* __restrict__ Ms, 
    int T, int N, int n_state
) {
    // Ms is shape (T, N, n_state, n_state)
    // alpha is shape (T + 1, N, n_state)
    // assumes blockDim = (N, 1, 1) and threadDim = (n_state, 1, 1)

    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= n_state) return;
    FLOAT u;
    for (int t = 0; t < T; t++) {
        int j = (t * N + bx) * n_state;
        u = MUL(Ms[(j + tx) * n_state], alpha[j]);
        for (int i = 1; i < n_state; i++) {
            u = SUM(u, MUL(Ms[(j + tx) * n_state + i], alpha[j + i]));
        }
        alpha[j + (N * n_state) + tx] = u;
        __syncthreads();
    }
  }