__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT b) {return a > b ? a : b;}
__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT b) {return a > b ? log1p(exp(b - a)) + a : log1p(exp(a - b)) + b;}
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}

extern "C" __global__ void sparse_Mv_scan(
    FLOAT* __restrict__ alpha,
    const FLOAT* __restrict__ Ms,  
    const int* __restrict__ idx,
    int T, int N, int C, int nz
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];
    
    FLOAT a = alpha[bx * C + tx];
    for (int t = 0; t < T; t++) {
        FLOAT *buf = smem + (t % 2) * blockDim.x;
        buf[tx] = a; __syncthreads();      
        int i = ((t * N + bx) * C) + tx;
        a = MUL(buf[idx[tx * nz]], Ms[i * nz]);
        for (int j = 1; j < nz; j++) {
            a = ADD(a, MUL(buf[idx[tx * nz + j]], Ms[i * nz + j]));
        }
        alpha[i + N * C] = a;
    }
}