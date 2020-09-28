__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT max_(FLOAT *s) {
    FLOAT mx = s[0];
    for (int j = 1; j < NZ; j++) {
        mx = mx > s[j] ? mx : s[j];
    }
    return mx;
}
__device__ __forceinline__ FLOAT logsumexp(FLOAT *s) {
    FLOAT mx = max_(s);
    FLOAT res = exp(s[0] - mx);
    for (int j = 1; j < NZ; j++) {
        res += exp(s[j] - mx);
    }
    return log(res) + mx;
}
    
extern "C" __global__ void logZ_fwd(
    FLOAT* __restrict__ logZ,
    FLOAT* __restrict__ Ms_grad,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ v0,
    const FLOAT* __restrict__ vT,
    const int* __restrict__ idx,
    int T, int N, int C
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x * K;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];
    
    FLOAT a[K];
    for (int k = 0; k < K; k++) {
        a[k] = v0[bx * C + tx + k]; 
    }
    __syncthreads();
    
    FLOAT s[NZ];
    for (int t = 0; t < T; t++) {
        FLOAT *buf = smem + (t % 2) * blockDim.x * K;
        for (int k = 0; k < K; k++) {
            buf[tx+k] = a[k];
        }
        __syncthreads();
        int i = (t * N + bx) * C * NZ;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NZ; j++) {
                s[j] = MUL(buf[idx[(tx + k) * NZ + j]], Ms[i + (tx + k) * NZ + j]);
                Ms_grad[i + (tx + k) * NZ + j] = s[j];
            }
            a[k] = SUM(s);        
        }
    }

    for (int k = 0; k < K; k++) {
        logZ[bx * C + tx + k] = MUL(a[k], vT[bx * C + tx + k]);
    }
}

extern "C" __global__ void logZ_bwd(
    FLOAT* __restrict__ betas,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ vT,
    const int* __restrict__ idx_T,
    int T, int N, int C
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x * K;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];
    
    FLOAT a[K];
    for (int k = 0; k < K; k++) {
        a[k] = vT[bx * C + tx + k];
        betas[(T * N + bx) * C + tx + k] = a[k];
    }
    __syncthreads();
    
    FLOAT s[NZ];
    for (int t = T - 1; t >= 0; t--) {
        FLOAT *buf = smem + (t % 2) * blockDim.x * K;
        for (int k = 0; k < K; k++) {
            buf[tx+k] = a[k];
        }
        __syncthreads(); 
        int i = (t * N + bx) * C;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NZ; j++) {
                int ix = idx_T[(tx + k) * NZ + j];
                s[j] = MUL(buf[ix / NZ], Ms[i * NZ + ix]);
            }            
            a[k] = SUM(s);
            betas[i + tx + k] = a[k];
        }        
    }
}
