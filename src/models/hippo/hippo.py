"""HiPPO [High-order Polynomial Projection Operators] initialization"""
import numpy as np
import torch
import torch.nn as nn

def transition(measure = 'legs', N=64, **kwargs):
    """
    computes HiPPO transition matricies

    Arguments:
        measure: 'legs' for scaled Legendre
        N: State dimension
    
    Returns: 
        A: [N, N] transition matrix
        B: [N, 1] input matrix
    """
    if measure == 'legs':
        return legs_matrix(N)
    else:
        return None
    
def legs_matrix(N):
    """
    HiPPO-LegS matrix for LRDs

    Uses scaled Legendre measure (optimal for learning LRD)
    """
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1

    # construct matrix, apply scaling, and return the HiPPO and Input matrices
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    
    # HiPPO matrix
    A = T @ M @ np.linalg.inv(T)
    
    # Input matrix
    B = np.sqrt(2 * q + 1)[:, None]
    
    return A, B

def nplr(measure = 'legs', N=64, **kwargs):
    """
    Normal-Plus-low-rank representation

    Returns matricies in NPLR form for efficiency
    """
    A, B = transition(measure=measure, N=N, **kwargs)

    # decompistion
    L, V = np.linalg.eig(A)
    L = L.astype(np.complex128)

    # low-rank correction
    P = np.conj(V)
    Q = V
    
    return L, P, Q, B

def hippo_init(nn_module, measure='legs', **kwargs):
    """
    Inits an S4 model with HiPPO

    Args:
        nn_module: S4 or S4D module to init
        measure: 'legs' for HiPPO-LegS
        **kwargs: Additional parameters
    """

    A, B = transition(measure=measure, N=nn_module.d_state, **kwargs)

    # initialize using hippo matrices
    with torch.no_grad():
        # handle different parameterizations
        if hasattr(nn_module, 'A'):
            # full matrix parameterization (legacy)
            nn_module.A.data = torch.from_numpy(A).float()
        
        if hasattr(nn_module, 'Lambda_re') and hasattr(nn_module, 'Lambda_im'):
            # diagonal/nplr parameterization (new s4 kernels)
            eigenvals = np.linalg.eigvals(A)
            Lambda_re = np.real(eigenvals)
            Lambda_im = np.imag(eigenvals)
            
            # use actual parameter size (for NPLR, this is d_state//2)
            param_N = nn_module.Lambda_re.shape[1]
            Lambda_re = Lambda_re[:param_N]
            Lambda_im = Lambda_im[:param_N]
            
            # repeat for all channels
            nn_module.Lambda_re.data = torch.from_numpy(Lambda_re).float().unsqueeze(0).repeat(nn_module.d_model, 1)
            nn_module.Lambda_im.data = torch.from_numpy(Lambda_im).float().unsqueeze(0).repeat(nn_module.d_model, 1)
        
        if hasattr(nn_module, 'A_log'):
            # legacy s4d log space initialization
            eigenvals = np.real(np.linalg.eigvals(A))
            nn_module.A_log.data = torch.from_numpy(eigenvals).unsqueeze(0).repeat(nn_module.d_model, 1)
        
        if hasattr(nn_module, 'B'):
            # input matrix
            B_tensor = torch.from_numpy(B).float()  # (N, 1)
            
            # use actual parameter size (for NPLR, this is d_state//2)
            param_N = nn_module.B.shape[1]
            B_tensor = B_tensor[:param_N, :]  # truncate if needed
            
            # handle different B shapes
            if len(nn_module.B.shape) == 3:
                # (d_model, d_state, 1) format
                if nn_module.B.dtype == torch.complex64 or nn_module.B.dtype == torch.complex128:
                    # complex parameterization (for NPLR)
                    # B_tensor is (param_N, 1), need to make it (d_model, param_N, 1)
                    B_tensor = B_tensor.squeeze(-1)  # (param_N,)
                    B_tensor = B_tensor.unsqueeze(0).unsqueeze(-1)  # (1, param_N, 1)
                    nn_module.B.data = B_tensor.repeat(nn_module.d_model, 1, 1).to(nn_module.B.dtype)
                else:
                    # real parameterization
                    nn_module.B.data = B_tensor.T.unsqueeze(-1).repeat(nn_module.d_model, 1, 1)
            else:
                # (d_model, d_state) format
                nn_module.B.data = B_tensor.T.repeat(nn_module.d_model, 1)

if __name__ == "__main__":
    # Test HiPPO initialization
    A, B = legs_matrix(N=16)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A eigenvalues: {np.real(np.linalg.eigvals(A))[:5]}")
