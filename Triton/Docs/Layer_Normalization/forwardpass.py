import torch

import triton
import triton.language as tl


@tirton.jit
def _layer_norm_fwd_pass(
    X,      # input pointer
    Y,      # output pointer
    W,      # weight pointer
    B,      # bias pointer
    Mean,   # mean pointer
    Rstd,   # inverse standard deviation pointer
    N,      # number of cols in 
    stride, # no. of stride to next row  
    eps,    # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Associate each program id to each row
    row = tl.program_id(axis=0)
    X += row * stride
    Y += row * stride

    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store mean and rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)