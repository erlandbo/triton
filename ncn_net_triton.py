import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F
import torch
import math
import time
from typing import Tuple


from utils import _take_slice_
from group_tensors import extract_group_index_


#torch.backends.cuda.matmul.allow_tf32 = False
#import torch
#torch.manual_seed(0)
#import random
#random.seed(0)
#import numpy as np
#np.random.seed(0)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def leaky_relu_grad(x):
    return tl.where(x >= 0, 1.0, 0.01)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def tanh_grad(x):
    # Tanh is just a scaled sigmoid
    t = tanh(x)
    return 1 - t * t



# @triton.autotune(
#     [
#         triton.Config(
#             {"BLOCK_X": BLOCK_SIZE_X},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for BLOCK_SIZE_X in [1, 2, 4, 8]
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["CTX_LEN",  "EMBED_DIM"],
# )
@triton.jit
def forward_kernel(
        # Pointers to matrices
        x_ptr, yi_ptr, ya_ptr, W_ptr, group_tensor_ptr,
        norm_alpha_ptr, norm_gamma_ptr, norm_beta_ptr,
        CTXLEN_OFFSETS,  # this could be int or pointer
        GROUP: tl.constexpr,
        # Matrix dimensions
        BATCH_SIZE: tl.constexpr, MAX_CTX_LEN: tl.constexpr, EMBED_DIM: tl.constexpr,
        # Meta-parameters
        ACTIVATION: tl.constexpr,
        ALPHA,
):
    
    offs_batch = tl.program_id(axis=0)
    offs_group = tl.program_id(axis=1)

    ctxlen = tl.load(CTXLEN_OFFSETS + offs_batch)
    #group_indices = tl.load(group_tensor_ptr + tl.arange(0, GROUP) + offs_group * GROUP, mask=tl.arange(0, GROUP) + offs_group * GROUP < ctxlen).to(tl.int32)
    group_indices_offsets = offs_batch * MAX_CTX_LEN + tl.arange(0, GROUP) + offs_group * GROUP
    group_indices = tl.load(group_tensor_ptr + group_indices_offsets, mask=tl.arange(0, GROUP) + offs_group * GROUP < ctxlen).to(tl.int32)

    xi_offsets = offs_batch * MAX_CTX_LEN * EMBED_DIM + group_indices[:, None] * EMBED_DIM + tl.arange(0, EMBED_DIM)[None, :]
    mask = offs_group * GROUP + tl.arange(0, GROUP)[:, None] < ctxlen + tl.arange(0, GROUP)[:, None]

    xi = tl.load(x_ptr + xi_offsets, mask=mask)
    xa = tl.zeros_like(xi)

    # Linear kernel
    offs_Wi = tl.arange(0, EMBED_DIM)
    offs_Wj = tl.arange(EMBED_DIM, 2*EMBED_DIM)
    Wi = tl.load(W_ptr + offs_Wi)
    Wj = tl.load(W_ptr + offs_Wj)

    # DyT
    norm_alpha_1 = tl.load(norm_alpha_ptr + 0)
    norm_gamma_1 = tl.load(norm_gamma_ptr + tl.arange(0, EMBED_DIM))
    norm_beta_1 = tl.load(norm_beta_ptr + tl.arange(0, EMBED_DIM))

    norm_alpha_2 = tl.load(norm_alpha_ptr + 1)
    norm_gamma_2 = tl.load(norm_gamma_ptr + tl.arange(EMBED_DIM, 2*EMBED_DIM))
    norm_beta_2 = tl.load(norm_beta_ptr + tl.arange(EMBED_DIM, 2*EMBED_DIM))

    #if offs_group * GROUP + GROUP < ctxlen:
    #    lo, hi = 0, GROUP
    #else:
    #    lo, hi = 0, ctxlen - offs_group * GROUP
    lo, hi = 0, GROUP

    for start_j in range(lo, hi):

        is_valid = _take_slice_(mask, 2, 1, start_j, GROUP, False)

        if tl.sum(is_valid) > 0.0:
            
            xj = _take_slice_(xi, 2, 1, start_j, GROUP, True).broadcast_to(xi.shape)
            
            #import pdb; pdb.set_trace()

            # tl.dot() only work for 2d or 3 matrices M, K, N >=16
            sim = tl.sum(xi * Wi[None, :], axis=1) + tl.sum(xj * Wj[None, :], axis=1)  # matrix-vector product (B,E)
            T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj

            T_norm = norm_gamma_1 * tanh(norm_alpha_1 * T) + norm_beta_1

            # optional: fused activation (while the data is in shared memory)
            if ACTIVATION == "leaky-relu":
                F = leaky_relu(T_norm)
            else:
                F = T_norm

            ya = xa + F
            
            ya_norm = norm_gamma_2 * tanh(norm_alpha_2 * ya) + norm_beta_2
            yi = xi + ya_norm

            xa = ya
            xi = yi

        else:
            pass
            # pass
    
    tl.store(yi_ptr + xi_offsets, xi, mask=mask) 
    tl.store(ya_ptr + xi_offsets, xa, mask=mask)



# @triton.autotune(
#     [
#         triton.Config(
#             {},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["EMBED_DIM"],
# )
@triton.jit
def backward_kernel(
        # Pointers to matrices
        x_ptr, xi_recon_ptr, xa_recon_ptr, yi_ptr, ya_ptr, dyi_ptr, dya_ptr, W_ptr, dW_ptr, dxi_ptr, dxa_ptr, group_tensor_ptr,
        norm_alpha_ptr, norm_gamma_ptr, norm_beta_ptr, dnorm_alpha_ptr, dnorm_gamma_ptr, dnorm_beta_ptr,
        CTXLEN_OFFSETS,  # this could be int or pointer
        GROUP: tl.constexpr,
        # Matrix dimensions
        BATCH_SIZE: tl.constexpr, MAX_CTX_LEN: tl.constexpr, EMBED_DIM: tl.constexpr,
        # Meta-parameters
        ACTIVATION: tl.constexpr,
        ALPHA,
):
    
    offs_batch = tl.program_id(axis=0)
    offs_group = tl.program_id(axis=1)

    # if offs_group == 33:
    #    import pdb; pdb.set_trace()

    ctxlen = tl.load(CTXLEN_OFFSETS + offs_batch)
    #group_indices = tl.load(group_tensor_ptr + tl.arange(0, GROUP) + offs_group * GROUP, mask=tl.arange(0, GROUP) + offs_group * GROUP < ctxlen).to(tl.int32)
    group_indices_offsets = offs_batch * MAX_CTX_LEN + tl.arange(0, GROUP) + offs_group * GROUP
    group_indices = tl.load(group_tensor_ptr + group_indices_offsets, mask=tl.arange(0, GROUP) + offs_group * GROUP < ctxlen).to(tl.int32)


    xi_offsets = offs_batch * MAX_CTX_LEN * EMBED_DIM + group_indices[:, None] * EMBED_DIM + tl.arange(0, EMBED_DIM)[None, :]
    mask = offs_group * GROUP + tl.arange(0, GROUP)[:, None] < ctxlen + tl.arange(0, GROUP)[:, None]

    # if offs_group == 33:
    #    import pdb; pdb.set_trace()

    yi = tl.load(yi_ptr + xi_offsets, mask=mask)
    ya = tl.load(ya_ptr + xi_offsets, mask=mask)

    dyi = tl.load(dyi_ptr + xi_offsets, mask=mask)
    dya = tl.load(dya_ptr + xi_offsets, mask=mask)

    # Linear kernel
    offs_Wi = tl.arange(0, EMBED_DIM)
    offs_Wj = tl.arange(EMBED_DIM, 2*EMBED_DIM)
    Wi = tl.load(W_ptr + offs_Wi)
    Wj = tl.load(W_ptr + offs_Wj)

    dWi = tl.zeros([EMBED_DIM], dtype=tl.float32)
    dWj = tl.zeros([EMBED_DIM], dtype=tl.float32)

    # DyT
    norm_alpha_1 = tl.load(norm_alpha_ptr + 0)
    norm_gamma_1 = tl.load(norm_gamma_ptr + tl.arange(0, EMBED_DIM))
    norm_beta_1 = tl.load(norm_beta_ptr + tl.arange(0, EMBED_DIM))

    norm_alpha_2 = tl.load(norm_alpha_ptr + 1)
    norm_gamma_2 = tl.load(norm_gamma_ptr + tl.arange(EMBED_DIM, 2*EMBED_DIM))
    norm_beta_2 = tl.load(norm_beta_ptr + tl.arange(EMBED_DIM, 2*EMBED_DIM))


    dalpha1 = tl.zeros([1], dtype=tl.float32)
    dgamma1 = tl.zeros([EMBED_DIM], dtype=tl.float32)
    dbeta1 = tl.zeros([EMBED_DIM], dtype=tl.float32)

    dalpha2 = tl.zeros([1], dtype=tl.float32)
    dgamma2 = tl.zeros([EMBED_DIM], dtype=tl.float32)
    dbeta2 = tl.zeros([EMBED_DIM], dtype=tl.float32)


    #if offs_group * GROUP + GROUP < ctxlen:
    #    lo, hi = 0, GROUP
    #else:
    #    lo, hi = 0, ctxlen - offs_group * GROUP
    lo, hi = 0, GROUP

    for start_j in range(hi-1, lo-1, -1):

        is_valid = _take_slice_(mask, 2, 1, start_j, GROUP, False)
        if tl.sum(is_valid) > 0.0:

            # Use reversibel trick
            ya_norm = norm_gamma_2 * tanh(norm_alpha_2 * ya) + norm_beta_2
            xi = yi - ya_norm

            xj = _take_slice_(xi, 2, 1, start_j, GROUP, True).broadcast_to(xi.shape)

            sim = tl.sum(xi * Wi[None, :], axis=1) + tl.sum(xj * Wj[None, :], axis=1)  # matrix-vector product (B,E)
            T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj

            T_norm = norm_gamma_1 * tanh(norm_alpha_1 * T) + norm_beta_1

            if ACTIVATION == "leaky-relu":
                F = leaky_relu(T_norm)
                grad_activation = leaky_relu_grad(T_norm)
            else:
                F = T_norm
                grad_activation = 1.0

            xa = ya - F

            ###############
            # backprop
            
            # dW
            grad_norm2 = (norm_gamma_2 * tanh_grad(norm_alpha_2 * ya) * norm_alpha_2) # (B,E)
            grad_norm1 = (norm_gamma_1 * tanh_grad(norm_alpha_1 * T) * norm_alpha_1)  # (B,E)
            
            dyi_dT = (grad_norm2 * grad_activation * grad_norm1 ) # (B,E)
            dya_dT = (grad_activation * grad_norm1 ) # (B,E)

            dWi_group_i = tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * xi * (1.0-ALPHA)
            dWi_group_a = tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * xi * (1.0-ALPHA)

            dWj_group_i = tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * xj * (1.0-ALPHA)
            dWj_group_a = tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * xj * (1.0-ALPHA)

            dWi_group = dWi_group_i + dWi_group_a
            dWj_group = dWj_group_i + dWj_group_a

            dWi += tl.sum(dWi_group, axis=0)
            dWj += tl.sum(dWj_group, axis=0)

            # Norm weights
            dyi_dgamma2 = tanh(norm_alpha_2 * ya)
            dyi_dalpha2 = norm_gamma_2 * tanh_grad(norm_alpha_2 * ya) * ya
            dyi_dbeta2 = tl.zeros_like(norm_beta_2) + 1.0

            dya_dgamma2 = 0.0
            dya_dalpha2 = 0.0
            dya_dbeta2 = 0.0

            dyi_dTnorm = (grad_norm2 * grad_activation) #[:,:,None] * eye
            dyi_dgamma1 = dyi_dTnorm * tanh(norm_alpha_1 * T)
            dyi_dalpha1 = dyi_dTnorm * (norm_gamma_1 * tanh_grad(norm_alpha_1 * T) * T)
            dyi_dbeta1 = dyi_dTnorm * (tl.zeros_like(norm_beta_1) + 1.0)

            dya_dTnorm = (grad_activation) #[:,:,None] * eye
            dya_dgamma1 = dya_dTnorm * tanh(norm_alpha_1 * T)
            dya_dalpha1 = dya_dTnorm * (norm_gamma_1 * tanh_grad(norm_alpha_1 * T) * T)
            dya_dbeta1 = dya_dTnorm * (tl.zeros_like(norm_beta_1) + 1.0)

            dgamma2_group = dyi * dyi_dgamma2 + dya * dya_dgamma2
            dalpha2_group = tl.sum(dyi * dyi_dalpha2, axis=-1) + tl.sum(dya * dya_dalpha2, axis=-1)
            dbeta2_group = dyi * dyi_dbeta2 + dya * dya_dbeta2

            dgamma1_group = dyi * dyi_dgamma1 + dya * dya_dgamma1
            dalpha1_group = tl.sum(dyi * dyi_dalpha1, axis=-1) + tl.sum(dya * dya_dalpha1, axis=-1)
            dbeta1_group = dyi * dyi_dbeta1 + dya * dya_dbeta1

            dgamma2 += tl.sum(dgamma2_group, axis=0)
            dalpha2 += tl.sum(dalpha2_group, axis=0)
            dbeta2 += tl.sum(dbeta2_group, axis=0)

            dgamma1 += tl.sum(dgamma1_group, axis=0)
            dalpha1 += tl.sum(dalpha1_group, axis=0)
            dbeta1 += tl.sum(dbeta1_group, axis=0)

            # xi xa
            dxi_i = dyi + ALPHA * dyi * dyi_dT + (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * Wi[None, :] )  # (B,E)
            dxi_a = ALPHA * dya * dya_dT + (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * Wi[None, :] ) # (B,E)
            dxi = dxi_i + dxi_a # (B,E)

            dxa_i = dyi * grad_norm2
            dxa_a = dya
            dxa = dxa_i + dxa_a # (B,E)

            # if i==j
            xj_mask = tl.arange(0, GROUP) == start_j
            #dxij_i = dyi + ALPHA * dyi * dyi_dT + (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * (Wi+Wj)[None, :]) + (1.0-ALPHA) * dyi * dyi_dT * sim[:, None]
            #dxij_a = ALPHA * dya * dya_dT + (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * (Wi+Wj)[None, :]) + (1.0-ALPHA) * dya * dya_dT * sim[:, None]
            dxij_i = (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * (Wj)[None, :]) + (1.0-ALPHA) * dyi * dyi_dT * sim[:, None]
            dxij_a = (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * (Wj)[None, :]) + (1.0-ALPHA) * dya * dya_dT * sim[:, None]
            dxij = dxij_i + dxij_a # (B,E)

            dxi += tl.where(xj_mask[:, None], dxij, 0.0)

            # if i!=j
            dxj_i = (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * Wj[None, :]) + (1.0-ALPHA) * dyi * dyi_dT * sim[:, None]
            dxj_a = (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * Wj[None, :]) + (1.0-ALPHA) * dya * dya_dT * sim[:, None]
            dxj_group = dxj_i + dxj_a # (B,E)

            dxj = tl.sum(tl.where((xj_mask==False)[:, None], dxj_group, 0.0), axis=0)
            dxi += tl.where(xj_mask[:, None], dxj[None, :], tl.zeros_like(dxi))

            ya = xa
            yi = xi
            dyi = dxi
            dya = dxa
        
        else:
            pass

    tl.store(dxi_ptr + xi_offsets, dyi, mask=mask) 
    tl.store(dxa_ptr + xi_offsets, dya, mask=mask) 
    ######
    tl.store(xi_recon_ptr + xi_offsets, yi, mask=mask) 
    tl.store(xa_recon_ptr + xi_offsets, ya, mask=mask) 

    num_groups = tl.cdiv(MAX_CTX_LEN, GROUP)
    #dWi_offsets = offs_batch * num_groups * 2*EMBED_DIM + offs_group * num_groups * EMBED_DIM + tl.arange(0, EMBED_DIM)
    #dWj_offsets = offs_batch * num_groups * 2*EMBED_DIM + offs_group * num_groups * EMBED_DIM + tl.arange(EMBED_DIM, 2*EMBED_DIM)
    #if offs_batch==0 and offs_group==7:
    #    import pdb; pdb.set_trace()
    
    dWi_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(0, EMBED_DIM) 
    dWj_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(EMBED_DIM, 2*EMBED_DIM)

    dgamma_1_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(0, EMBED_DIM) 
    dgamma_2_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(EMBED_DIM, 2*EMBED_DIM)
    dbeta_1_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(0, EMBED_DIM) 
    dbeta_2_offsets = offs_batch * (num_groups * 2*EMBED_DIM) + offs_group * (2*EMBED_DIM) + tl.arange(EMBED_DIM, 2*EMBED_DIM)
    dalpha_1_offsets = offs_batch * (num_groups * 2) + offs_group * (2) + tl.arange(0, 1) 
    dalpha_2_offsets = offs_batch * (num_groups * 2) + offs_group * (2) + tl.arange(1, 2)

    
    tl.store(dW_ptr + dWi_offsets, dWi)
    tl.store(dW_ptr + dWj_offsets, dWj)

    tl.store(dnorm_alpha_ptr + dalpha_1_offsets, dalpha1)
    tl.store(dnorm_alpha_ptr + dalpha_2_offsets, dalpha2)
    tl.store(dnorm_beta_ptr + dbeta_1_offsets, dbeta1)
    tl.store(dnorm_beta_ptr + dbeta_2_offsets, dbeta2)
    tl.store(dnorm_gamma_ptr + dgamma_1_offsets, dgamma1)
    tl.store(dnorm_gamma_ptr + dgamma_2_offsets, dgamma2)



class FusedLinearChunkNCNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, norm_alpha, norm_beta, norm_gamma, group_tensor, seqlen_offsets, ALPHA, ACTIVATION):
        # shape constraints
        BATCH_SIZE, CTX_LEN, EMBED_DIM = x.shape
        xa = torch.zeros_like(x)
        yi = torch.empty_like(x)
        ya = torch.empty_like(xa)

        GROUP = 64
        NUM_GROUPS = triton.cdiv(CTX_LEN, GROUP)
        ctx_indices = torch.arange(CTX_LEN)

        grid = lambda META: (BATCH_SIZE, triton.cdiv(CTX_LEN, GROUP))
        forward_kernel[grid](
            x_ptr=x, yi_ptr=yi, ya_ptr=ya, W_ptr=W, group_tensor_ptr=group_tensor,
            norm_alpha_ptr=norm_alpha, norm_beta_ptr=norm_beta, norm_gamma_ptr=norm_gamma, 
            BATCH_SIZE=BATCH_SIZE, MAX_CTX_LEN=CTX_LEN, EMBED_DIM=EMBED_DIM,
            ACTIVATION=ACTIVATION,
            ALPHA=ALPHA,
            CTXLEN_OFFSETS=seqlen_offsets,
            GROUP=GROUP,
        )

        ctx.save_for_backward(x, W, group_tensor, ctx_indices, yi, ya, norm_alpha, norm_beta, norm_gamma, )
        ctx.ALPHA = ALPHA
        ctx.GROUP = GROUP
        ctx.ACTIVATION = ACTIVATION
        ctx.CTXLEN_OFFSETS = seqlen_offsets
        return yi, ya


    @staticmethod
    def backward(ctx, dyi, dya):
        x, W, group_tensor, ctx_indices, yi, ya, norm_alpha, norm_beta, norm_gamma  = ctx.saved_tensors
        BATCH_SIZE, CTX_LEN, EMBED_DIM = x.shape
        NUM_GROUPS = triton.cdiv(CTX_LEN, ctx.GROUP)

        dxi = torch.zeros_like(dyi)
        dxa = torch.zeros_like(dya)

        xi_recon = torch.zeros_like(dyi)
        xa_recon = torch.zeros_like(dya)

        dW = torch.zeros((BATCH_SIZE, NUM_GROUPS, 2*EMBED_DIM), dtype=W.dtype, device=W.device)
        dnorm_alpha = torch.zeros((BATCH_SIZE, NUM_GROUPS, 2), dtype=W.dtype, device=W.device)
        dnorm_beta = torch.zeros((BATCH_SIZE, NUM_GROUPS, 2*EMBED_DIM), dtype=W.dtype, device=W.device)
        dnorm_gamma = torch.zeros((BATCH_SIZE, NUM_GROUPS, 2*EMBED_DIM), dtype=W.dtype, device=W.device)


        #grid = lambda META: (BATCH_SIZE, triton.cdiv(CTX_LEN, META["GROUP"]))
        grid = lambda META: (BATCH_SIZE, triton.cdiv(CTX_LEN, ctx.GROUP))

        backward_kernel[grid](
            x_ptr=x, xi_recon_ptr=xi_recon, xa_recon_ptr=xa_recon, yi_ptr=yi, ya_ptr=ya, dyi_ptr=dyi, dya_ptr=dya, 
            W_ptr=W, dW_ptr=dW, dxi_ptr=dxi, dxa_ptr=dxa, group_tensor_ptr=group_tensor,
            norm_alpha_ptr=norm_alpha, norm_beta_ptr=norm_beta, norm_gamma_ptr=norm_gamma, 
            dnorm_alpha_ptr=dnorm_alpha, dnorm_beta_ptr=dnorm_beta, dnorm_gamma_ptr=dnorm_gamma, 
            BATCH_SIZE=BATCH_SIZE, MAX_CTX_LEN=CTX_LEN, EMBED_DIM=EMBED_DIM,
            ACTIVATION=ctx.ACTIVATION,
            ALPHA=ctx.ALPHA,
            CTXLEN_OFFSETS=ctx.CTXLEN_OFFSETS,
            GROUP=ctx.GROUP,
        )
        #import pdb; pdb.set_trace()
        sum_dW = torch.sum(dW, dim=(0, 1))
        sum_dnorm_alpha = torch.sum(dnorm_alpha, dim=(0, 1))
        sum_dnorm_beta = torch.sum(dnorm_beta, dim=(0, 1))
        sum_dnorm_gamma = torch.sum(dnorm_gamma, dim=(0, 1))
        return dxi, sum_dW, sum_dnorm_alpha, sum_dnorm_beta, sum_dnorm_gamma, None, None, None, None



def fused_chunk_linear_ncn(
    x: torch.Tensor, 
    W: torch.Tensor, 
    norm_alpha: torch.Tensor, 
    norm_beta: torch.Tensor, 
    norm_gamma: torch.Tensor, 
    group_tensor: torch.Tensor, 
    seqlen_offsets: torch.Tensor, 
    ALPHA: float, 
    ACTIVATION: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yi, ya = FusedLinearChunkNCNFunction.apply(
        x, 
        W, 
        norm_alpha, 
        norm_beta, 
        norm_gamma, 
        group_tensor, 
        seqlen_offsets, 
        ALPHA, 
        ACTIVATION
    )
    return yi, ya



class NCNLinearKernel(nn.Module):
    def __init__(
        self,
        alpha: float = 0.9,
        activation: str = "leaky-relu",
        dmodel: int = 64,
        num_layers: int = 1,
        norm_alpha_init: float = 1.0,
        norm_beta_init: float = 1.0,
        norm_gamma_init: float = 1.0,
    ):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(2*dmodel).normal_(mean=0.0, std=1/(2.0 * dmodel)), requires_grad=True).cuda() for _ in range(num_layers)])
        self.norm_alphas = nn.ParameterList([nn.Parameter(torch.ones(2) * norm_alpha_init, requires_grad=True).cuda() for _ in range(num_layers)])
        self.norm_betas = nn.ParameterList([nn.Parameter(torch.ones(2*dmodel) * norm_beta_init, requires_grad=True).cuda() for _ in range(num_layers)])
        self.norm_gammas = nn.ParameterList([nn.Parameter(torch.ones(2*dmodel) * norm_gamma_init, requires_grad=True).cuda() for _ in range(num_layers)])
        self.alpha = alpha
        self.activation = activation
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        #index_within_group_tensor, group_number_tensor = extract_group_index_()
        BATCH_SIZE, MAX_CTX_LEN, EMBED_DIM = x.shape
        ctxlen_tensor =  mask.sum(-1).to(torch.int32)

        for l in range(self.num_layers):
            #import pdb; pdb.set_trace()
            #group_number_tensor = torch.stack([F.pad(torch.arange(0, ctxlen), pad=(0, MAX_CTX_LEN - ctxlen.item())) for ctxlen in ctxlen_tensor]).to(torch.int32).cuda()
            group_number_tensor = torch.stack([F.pad(torch.randperm(ctxlen), pad=(0, MAX_CTX_LEN - ctxlen.item())) for ctxlen in ctxlen_tensor]).to(torch.int32).cuda()
            x, xa = fused_chunk_linear_ncn(
                x, 
                self.weights[l], 
                self.norm_alphas[l], 
                self.norm_betas[l], 
                self.norm_gammas[l], 
                group_number_tensor, 
                ctxlen_tensor, 
                self.alpha, 
                self.activation, 
            )
        return x

if __name__ == "__main__":

    BATCH_SIZE, SEQ_LEN, EMBED_DIM = 8, 4096, 64

    dtype=torch.float32

    ncnnet = NCNLinearKernel(
        alpha=0.9,
        activation="leaky-relu",
        dmodel=EMBED_DIM,
        num_layers=3,
        norm_alpha_init=1.0,
        norm_beta_init=1.0,
        norm_gamma_init=1.0
    ).cuda()

    x = (
        torch.ones((BATCH_SIZE, SEQ_LEN, EMBED_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    print(x[0])

    MASK = torch.ones((BATCH_SIZE, SEQ_LEN), device="cuda")
    MASK = torch.tril(MASK)
    print(MASK)

    dyi = torch.randn_like(x)
    dya = torch.randn_like(x)

    start_time = time.time()

    y = ncnnet(x, MASK)
    y.backward(dyi)

    print(time.time() - start_time)
