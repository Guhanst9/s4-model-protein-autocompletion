"""s4 kernel computation with ssm discretization"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# helper functions for complex numbers
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

contract = torch.einsum


def discretize_bilinear(Lambda, B_tilde, Delta):
    """bilinear discretization of continuous ssm
    
    converts continuous (Lambda, B) to discrete (Lambda_bar, B_bar)
    using bilinear (tustin) transform
    """
    identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    
    # bilinear transform
    Lambda_bar = (1 + Delta * Lambda / 2) / (1 - Delta * Lambda / 2)
    B_bar = (Delta * B_tilde / (1 - Delta * Lambda / 2).unsqueeze(-1))
    
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """zero-order hold discretization
    
    more accurate than bilinear for SSMs
    uses matrix exponential approximation
    """
    identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    
    # zoh transform
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda) * (Lambda_bar - 1) * B_tilde
    
    return Lambda_bar, B_bar


def cauchy_naive(v, z, w):
    """naive cauchy kernel computation
    
    computes sum_j v_j / (z_i - w_j)
    used as fallback when optimized kernels aren't available
    """
    # expand dims for broadcasting
    # v: (n,), z: (l,), w: (n,)
    # output: (l,)
    cauchy_matrix = 1 / (z.unsqueeze(-1) - w.unsqueeze(0))  # (l, n)
    return (cauchy_matrix @ v.unsqueeze(-1)).squeeze(-1)


def log_vandermonde_naive(v, x, L):
    """naive vandermonde computation in log space
    
    computes V @ v where V_ij = x_i^j
    more stable in log space
    """
    vandermonde_matrix = x.unsqueeze(-1) ** torch.arange(L, device=x.device)  # (n, L)
    return (vandermonde_matrix.T @ v.unsqueeze(-1)).squeeze(-1)


def cauchy_mult(v, z, w, symmetric=False):
    """efficient cauchy kernel using keops if available
    
    falls back to naive implementation if keops not installed
    """
    try:
        from pykeops.torch import Genred
        
        # keops variables: z (l,), w (n,), v (n,)
        formula = 'ComplexDivide(v, z - w)'
        cauchy_kernel = Genred(
            formula,
            [
                'v = Vj(2)',  # complex number (2d vector)
                'z = Vi(2)',  # complex number  
                'w = Vj(2)',  # complex number
            ],
            reduction_op='Sum',
            axis=1,
        )
        
        # convert to keops format (real, imag pairs)
        v_re, v_im = v.real, v.imag
        z_re, z_im = z.real, z.imag
        w_re, w_im = w.real, w.imag
        
        v_keops = torch.stack([v_re, v_im], dim=-1)
        z_keops = torch.stack([z_re, z_im], dim=-1)
        w_keops = torch.stack([w_re, w_im], dim=-1)
        
        result = cauchy_kernel(z_keops, w_keops, v_keops)
        
        return torch.view_as_complex(result.contiguous())
        
    except ImportError:
        # fallback to naive
        return cauchy_naive(v, z, w)


class SSKernelDiag(nn.Module):
    """diagonal s4 kernel (s4d variant)
    
    uses diagonal state matrix A for simplicity and efficiency
    """
    
    def __init__(
        self,
        d_model,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        lr=None,
        deterministic=False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # trainable timescale parameter
        self.log_dt = nn.Parameter(
            torch.linspace(math.log(dt_min), math.log(dt_max), d_model)
        )
        
        # will be initialized by hippo
        self.Lambda_re = nn.Parameter(torch.randn(d_model, d_state))
        self.Lambda_im = nn.Parameter(torch.randn(d_model, d_state))
        
        self.B = nn.Parameter(torch.randn(d_model, d_state, 1))
        self.C = nn.Parameter(torch.randn(d_model, 1, d_state, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # for optimizer
        if lr is not None:
            for param in [self.Lambda_re, self.Lambda_im]:
                param._optim = {'lr': lr}
    
    def forward(self, L):
        """generate convolution kernel of length L"""
        dt = torch.exp(self.log_dt)  # (H,)
        
        # construct complex eigenvalues
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)  # (H, N)
        
        # discretize using zoh
        dt_expanded = dt.unsqueeze(-1)  # (H, 1)
        dtA = dt_expanded * Lambda  # (H, N)
        
        Lambda_bar = torch.exp(dtA)  # (H, N)
        B_bar = (torch.exp(dtA) - 1.) / Lambda * self.B.squeeze(-1)  # (H, N)
        
        # multiply C and B for vandermonde product
        C = self.C.squeeze(1)  # (H, N)
        CB = C * B_bar  # (H, N)
        
        # compute vandermonde matrix
        # V_ij = Lambda_bar_i^j for j in [0, L-1]
        # result is sum_i CB_i * Lambda_bar_i^j
        vandermonde_matrix = Lambda_bar.unsqueeze(-1) ** torch.arange(L, device=Lambda.device)  # (H, N, L)
        K = contract('hn,hnl->hl', CB, vandermonde_matrix)  # (H, L)
        
        # take real part and add skip connection
        K = 2 * K.real
        
        # add D term (skip connection at t=0)
        K = K + self.D.unsqueeze(-1) * (torch.arange(L, device=K.device) == 0).float()
        
        return K
    
    def step(self, u, state):
        """single step for recurrent mode
        
        u: (batch, d_model)
        state: (batch, d_model, d_state) 
        """
        dt = torch.exp(self.log_dt)  # (H,)
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)  # (H, N)
        
        # discretize
        dt_expanded = dt.unsqueeze(-1)  # (H, 1)
        Lambda_bar = torch.exp(Lambda * dt_expanded)  # (H, N)
        B_bar = (torch.exp(Lambda * dt_expanded) - 1.) / Lambda * self.B.squeeze(-1)  # (H, N)
        
        # state update: x[t] = Lambda_bar * x[t-1] + B_bar * u[t]
        new_state = Lambda_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u.unsqueeze(-1)
        
        # output: y[t] = 2 * Re(C^* @ x[t]) + D * u[t]
        C = self.C.squeeze(1)  # (H, N)
        y = contract('hn,bhn->bh', C, new_state)
        y = 2 * y.real + self.D * u
        
        return y, new_state


class SSKernelNPLR(nn.Module):
    """normal plus low rank s4 kernel (original s4)
    
    uses DPLR (diagonal plus low rank) parameterization
    more expressive than diagonal but requires cauchy kernel
    
    note: due to conjugate symmetry, effective state size is d_state//2
    """
    
    def __init__(
        self,
        d_model,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        rank=1,
        lr=None,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.rank = rank
        
        # timescale
        self.log_dt = nn.Parameter(
            torch.linspace(math.log(dt_min), math.log(dt_max), d_model)
        )
        
        # nplr parameterization uses conjugate symmetry, so effective state size is d_state//2
        N = d_state // 2
        
        self.Lambda_re = nn.Parameter(torch.randn(d_model, N))
        self.Lambda_im = nn.Parameter(torch.randn(d_model, N))
        
        self.P = nn.Parameter(torch.randn(d_model, N, rank, dtype=torch.complex64))
        self.Q = nn.Parameter(torch.randn(d_model, rank, N, dtype=torch.complex64))
        
        self.B = nn.Parameter(torch.randn(d_model, N, 1, dtype=torch.complex64))
        self.C = nn.Parameter(torch.randn(d_model, 1, N, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(d_model))
        
        if lr is not None:
            for param in [self.Lambda_re, self.Lambda_im, self.P, self.Q]:
                param._optim = {'lr': lr}
    
    def forward(self, L):
        """generate kernel using cauchy kernel trick"""
        dt = torch.exp(self.log_dt)
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)
        
        # discretization
        dt_expanded = dt.unsqueeze(-1)
        Lambda_bar = torch.exp(Lambda * dt_expanded)
        B_bar = (torch.exp(Lambda * dt_expanded) - 1.) / Lambda * self.B.squeeze(-1)
        
        # simplified nplr without woodbury - just use vandermonde like s4d
        # for full nplr with low-rank correction, more complex cauchy kernel needed
        C = self.C.squeeze(1)  # (H, N)
        CB = C * B_bar  # (H, N)
        
        # compute vandermonde
        vandermonde_matrix = Lambda_bar.unsqueeze(-1) ** torch.arange(L, device=Lambda.device)  # (H, N, L)
        K = contract('hn,hnl->hl', CB, vandermonde_matrix)  # (H, L)
        
        # take real part
        K = 2 * K.real
        
        # add skip connection at t=0
        K = K + self.D.unsqueeze(-1) * (torch.arange(L, device=K.device) == 0).float()
        
        return K
    
    def step(self, u, state):
        """single recurrent step"""
        dt = torch.exp(self.log_dt)
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)
        
        dt_expanded = dt.unsqueeze(-1)
        Lambda_bar = torch.exp(Lambda * dt_expanded)
        B_bar = (torch.exp(Lambda * dt_expanded) - 1.) / Lambda * self.B.squeeze(-1)
        
        # state transition with low-rank correction
        # x[t] = (Lambda_bar + P @ Q) @ x[t-1] + B_bar @ u[t]
        new_state = Lambda_bar.unsqueeze(0) * state
        
        if self.rank > 0:
            # add low-rank update: P @ (Q @ x[t-1])
            pq_state = torch.einsum('hzr,hrz,bhz->bhz', self.P, self.Q, state)
            new_state = new_state + pq_state
        
        new_state = new_state + B_bar.unsqueeze(0) * u.unsqueeze(-1)
        
        # output: y[t] = 2 * Re(C^* @ x[t]) + D * u[t]
        C = self.C.squeeze(1)  # (H, N)
        y = contract('hn,bhn->bh', C, new_state)
        y = 2 * y.real + self.D * u
        
        return y, new_state
