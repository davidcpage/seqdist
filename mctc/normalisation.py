# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/04_Normalisation.ipynb (unless otherwise specified).

__all__ = ['device', 'generate_test_example', 'logZ_py', 'fused_batch_Mv', 'LogZ', 'LogZViterbi', 'cupy_funcs', 'logz',
           'logz_viterbi']

# Cell
import numpy as np
import cupy as cp
import torch
from .utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Cell
def generate_test_example(T, N, n_state, dtype=torch.float):
    return torch.rand((T, N, n_state, n_state), device=device, dtype=dtype, requires_grad=True)

# Cell
import torch

def _rescale(M):
    #T, N, n_state, n_state = M.shape
    Z = M.sum((2, 3), keepdim=True) / M.size(3)
    logZ = torch.log(Z).sum(0).reshape(-1)
    return M / Z, logZ

@torch.jit.script
def logZ_py(M, alpha_0):
    M, logZ = _rescale(M)
    T, N, n_state, _ = M.shape
    alpha = alpha_0.unsqueeze(2)
    for i, M_t in enumerate(M.unbind(0)):
        alpha = M_t.bmm(alpha)
        if i % 32 == (T - 1) % 32:
            z = alpha.sum(1, keepdim=True)
            alpha = alpha/z
            logZ += torch.log(z.squeeze())
    return logZ

# Cell
from .ctc import semiring, Log, Max
from functools import partial

cupy_funcs = {
    (torch.float32, Log): load_cupy_func('cuda/fused_bmv.cu', 'fwd', FLOAT='float', MUL='add', SUM='logsumexp2'),
    (torch.float64, Log): load_cupy_func('cuda/fused_bmv.cu', 'fwd', FLOAT='double', MUL='add', SUM='logsumexp2'),
    (torch.float32, Max): load_cupy_func('cuda/fused_bmv.cu', 'fwd', FLOAT='float',  MUL='add', SUM='max2'),
    (torch.float64, Max): load_cupy_func('cuda/fused_bmv.cu', 'fwd', FLOAT='double', MUL='add', SUM='max2'),
}

def fused_batch_Mv(Ms, alpha_0, S:semiring=Log):
    T, N, n_state, _ = Ms.shape
    alpha = Ms.new_empty((T + 1, N, n_state))
    alpha[0] = alpha_0
    with cp.cuda.Device(Ms.device.index):
        cupy_funcs[(Ms.dtype, S)](
            grid=(N, 1, 1),
            block=(n_state, 1, 1),
            args=(alpha.data_ptr(), Ms.contiguous().data_ptr(), T, N, n_state)
        )
    return alpha

def _logz_fwd(ctx, Ms, alpha_0, beta_T, S:semiring=Log):
    alpha = fused_batch_Mv(Ms, alpha_0, S)
    ctx.save_for_backward(Ms, alpha, beta_T)
    return S.sum(S.mul(alpha[-1], beta_T), dim=1)

def _logz_bwd(ctx, g, S:semiring=Log):
    Ms, alpha, beta_T = ctx.saved_tensors
    T, N, n_state, _ = Ms.shape
    beta = fused_batch_Mv(Ms.transpose(2, 3).flip(0), beta_T)
    Ms_grad = S.mul(S.mul(Ms, alpha[:-1,:,None,:]), (beta[:-1, :, :, None]).flip(0))
    Ms_grad = S.dsum(Ms_grad.reshape(T, N, -1), dim=2).reshape(T, N, n_state, n_state)
    return Ms_grad * g[None, :, None, None], None, None, None

class LogZ(torch.autograd.Function):
    forward = staticmethod(_logz_fwd)
    backward = staticmethod(_logz_bwd)

class LogZViterbi(torch.autograd.Function):
    forward = staticmethod(partial(_logz_fwd, S=Max))
    backward = staticmethod(partial(_logz_bwd, S=Max))

logz = LogZ.apply
logz_viterbi = LogZViterbi.apply