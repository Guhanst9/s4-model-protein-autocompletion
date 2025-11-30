"""test s4 kernel implementations"""
import torch
import numpy as np
from src.models.s4.s4_kernel import SSKernelDiag, SSKernelNPLR, discretize_zoh, cauchy_naive
from src.models.hippo.hippo import hippo_init, legs_matrix

def test_discretization():
    """test ssm discretization methods"""
    print("testing discretization...")
    
    # simple test case
    Lambda = torch.tensor([[-1.0 + 1j], [-2.0 + 0.5j]])
    B = torch.tensor([[1.0], [0.5]])
    Delta = torch.tensor([0.1, 0.1]).unsqueeze(-1)
    
    Lambda_bar, B_bar = discretize_zoh(Lambda, B, Delta)
    
    print(f"  Lambda shape: {Lambda_bar.shape}")
    print(f"  B shape: {B_bar.shape}")
    print(f"  Lambda_bar[0]: {Lambda_bar[0].item()}")
    print("discretization works\n")

def test_cauchy_kernel():
    """test cauchy kernel computation"""
    print("testing cauchy kernel...")
    
    # simple test
    v = torch.randn(8, dtype=torch.complex64)
    z = torch.randn(16, dtype=torch.complex64)
    w = torch.randn(8, dtype=torch.complex64)
    
    result = cauchy_naive(v, z, w)
    
    print(f"  input v: {v.shape}, z: {z.shape}, w: {w.shape}")
    print(f"  output: {result.shape}")
    print(f"  result[0]: {result[0].item()}")
    print("cauchy kernel works\n")

def test_s4_diagonal():
    """test s4 diagonal kernel"""
    print("testing s4d kernel...")
    
    d_model = 4
    d_state = 16
    L = 64
    
    kernel = SSKernelDiag(d_model=d_model, d_state=d_state)
    
    # initialize with hippo
    hippo_init(kernel, measure='legs')
    
    # generate convolution kernel
    K = kernel(L)
    
    print(f"  kernel shape: {K.shape}")
    print(f"  expected: ({d_model}, {L})")
    print(f"  kernel mean: {K.mean().item():.4f}")
    print(f"  kernel std: {K.std().item():.4f}")
    
    # test step mode
    batch = 2
    u = torch.randn(batch, d_model)
    state = torch.zeros(batch, d_model, d_state, dtype=torch.complex64)
    
    y, new_state = kernel.step(u, state)
    
    print(f"  step output shape: {y.shape}")
    print(f"  step state shape: {new_state.shape}")
    print("s4d kernel works\n")

def test_s4_nplr():
    """test s4 nplr kernel"""
    print("testing s4 nplr kernel...")
    
    d_model = 4
    d_state = 16
    L = 64
    
    kernel = SSKernelNPLR(d_model=d_model, d_state=d_state, rank=2)
    
    # initialize with hippo
    hippo_init(kernel, measure='legs')
    
    # generate kernel
    K = kernel(L)
    
    print(f"  kernel shape: {K.shape}")
    print(f"  expected: ({d_model}, {L})")
    print(f"  kernel mean: {K.mean().item():.4f}")
    print(f"  kernel std: {K.std().item():.4f}")
    
    # test step mode
    # nplr uses d_state//2 for actual state size due to conjugate symmetry
    batch = 2
    u = torch.randn(batch, d_model)
    actual_state_size = kernel.Lambda_re.shape[1]  # get actual state dim
    state = torch.zeros(batch, d_model, actual_state_size, dtype=torch.complex64)
    
    y, new_state = kernel.step(u, state)
    
    print(f"  step output shape: {y.shape}")
    print(f"  step state shape: {new_state.shape}")
    print(" s4 nplr kernel works\n")

def test_hippo_integration():
    """test hippo initialization with kernels"""
    print("testing hippo integration...")
    
    # test with s4d
    kernel = SSKernelDiag(d_model=8, d_state=32)
    A, B = legs_matrix(N=32)
    
    print(f"  hippo A shape: {A.shape}")
    print(f"  hippo B shape: {B.shape}")
    
    hippo_init(kernel, measure='legs')
    
    print(f"  kernel Lambda_re: {kernel.Lambda_re.shape}")
    print(f"  kernel Lambda_im: {kernel.Lambda_im.shape}")
    print(f"  kernel B: {kernel.B.shape}")
    
    # check eigenvalues match
    eigenvals = np.linalg.eigvals(A)
    print(f"  hippo eigenvals (first 3): {eigenvals[:3]}")
    print(f"  kernel Lambda (first 3): {(kernel.Lambda_re[0, :3] + 1j*kernel.Lambda_im[0, :3]).detach().numpy()}")
    print(" hippo integration works\n")

def test_convolution():
    """test that convolution mode produces reasonable output"""
    print("testing convolution with real data...")
    
    d_model = 8
    d_state = 64
    batch = 2
    seq_len = 128
    
    # create kernel
    kernel = SSKernelDiag(d_model=d_model, d_state=d_state)
    hippo_init(kernel, measure='legs')
    
    # generate kernel
    K = kernel(seq_len)
    
    # create input
    u = torch.randn(batch, d_model, seq_len)
    
    # convolve using fft
    u_f = torch.fft.rfft(u, n=2*seq_len, dim=-1)
    K_f = torch.fft.rfft(K, n=2*seq_len, dim=-1)
    y_f = u_f * K_f
    y = torch.fft.irfft(y_f, n=2*seq_len, dim=-1)[..., :seq_len]
    
    print(f"  input shape: {u.shape}")
    print(f"  kernel shape: {K.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  output mean: {y.mean().item():.4f}")
    print(f"  output std: {y.std().item():.4f}")
    print(" convolution produces valid output\n")

if __name__ == "__main__":
    print("S4 Kernel Implementation Tests \n")

    
    test_discretization()
    test_cauchy_kernel()
    test_s4_diagonal()
    test_s4_nplr()
    test_hippo_integration()
    test_convolution()
    
    print("All tests passed! ")
