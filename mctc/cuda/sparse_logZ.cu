__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT b) {return a > b ? a : b;}
__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT b) {return a > b ? log1p(exp(b - a)) + a : log1p(exp(a - b)) + b;}
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}

extern "C" __global__ void logZ_fwd_bwd(
    FLOAT* __restrict__ logZ,
    FLOAT* __restrict__ Ms_grad,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ Ms_T,
    const FLOAT* __restrict__ v0,
    const FLOAT* __restrict__ vT,
    const int* __restrict__ idx,
    const int* __restrict__ idx_T,
    int T, int N, int C
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];
    
    FLOAT a = v0[bx * C + tx];
    FLOAT tmp;
    for (int t = 0; t < T; t++) {
        FLOAT *buf = smem + (t % 2) * blockDim.x;
        buf[tx] = a; __syncthreads();      
        int i = (((t * N + bx) * C) + tx) * NZ;
        a = MUL(buf[idx[tx * NZ]], Ms[i]);
        Ms_grad[i] = a;
        for (int j = 1; j < NZ; j++) {
            tmp = MUL(buf[idx[tx * NZ + j]], Ms[i + j]);
            Ms_grad[i + j] = tmp;
            a = ADD(a, tmp);
        }
    }

    FLOAT b = vT[bx * C + tx];
    logZ[bx * C + tx] = MUL(a, b);
    __syncthreads();

    for (int t = T - 1; t >= 0; t--) {
        FLOAT *buf = smem + (t % 2) * blockDim.x;
        buf[tx] = b; __syncthreads(); 
        int i = (((t * N + bx) * C) + tx) * NZ;
        for (int j = 0; j < NZ; j++) {
            Ms_grad[i + j] = MUL(Ms_grad[i + j], b);
        }
        b = MUL(buf[idx_T[tx * NZ]], Ms_T[i]);
        for (int j = 1; j < NZ; j++) {
            b = ADD(b, MUL(buf[idx_T[tx * NZ + j]], Ms_T[i + j]));
        }
    }
}