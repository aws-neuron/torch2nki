import torch
import torch.nn as nn
import torch_xla
from torch_xla.core import xla_model as xm
import os

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from matrixops_nki_kernels import (
    nki_linalg_qr, nki_linalg_svd, nki_linalg_inv, nki_linalg_pinv,
    nki_linalg_matrix_norm, nki_linalg_vector_norm, nki_linalg_cross,
    nki_linalg_outer, nki_linalg_tensordot, nki_linalg_eigh, nki_linalg_eig,
    nki_linalg_slogdet, nki_linalg_solve, nki_linalg_lstsq,
    nki_linalg_cholesky, nki_linalg_lu, nki_linalg_ldl_factor,
    nki_linalg_triangular_solve
)

def test_torch_linalg_qr(device):
    """Test QR decomposition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match (within tolerance), 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    Q_nki, R_nki = nki_linalg_qr(A)
    Q_torch, R_torch = torch.linalg.qr(A)
    print("Checking correctness of QR decomposition...")
    match = torch.allclose(torch.matmul(Q_torch, R_torch), A, atol=1e-2, rtol=1e-2) and \
            torch.allclose(torch.matmul(Q_nki, R_nki), A, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_svd(device):
    """Test SVD decomposition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if singular values match, 0 otherwise.
    """
    A = torch.rand((8, 6), dtype=torch.bfloat16, device=device)
    U_nki, S_nki, Vh_nki = nki_linalg_svd(A)
    U_torch, S_torch, Vh_torch = torch.linalg.svd(A)
    print("Checking correctness of SVD decomposition (singular values)...")
    match = torch.allclose(S_torch, S_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_inv(device):
    """Test matrix inverse between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if inverses match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device) + torch.eye(8, dtype=torch.bfloat16, device=device)*0.5
    inv_nki = nki_linalg_inv(A)
    inv_torch = torch.linalg.inv(A)
    print("Checking correctness of matrix inverse...")
    match = torch.allclose(inv_torch, inv_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_pinv(device):
    """Test pseudo-inverse between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if pseudo-inverses match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    pinv_nki = nki_linalg_pinv(A)
    pinv_torch = torch.linalg.pinv(A)
    print("Checking correctness of pseudo-inverse...")
    match = torch.allclose(pinv_torch, pinv_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_matrix_norm(device):
    """Test matrix norm computation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if norms match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    norm_nki = nki_linalg_matrix_norm(A, ord='fro', dim=(-2, -1))
    norm_torch = torch.linalg.matrix_norm(A, ord='fro', dim=(-2, -1))
    print("Checking correctness of matrix norm...")
    match = torch.allclose(norm_torch, norm_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_vector_norm(device):
    """Test vector norm computation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if norms match, 0 otherwise.
    """
    A = torch.rand((10, 5), dtype=torch.bfloat16, device=device)
    norm_nki = nki_linalg_vector_norm(A, ord=2, dim=1)
    norm_torch = torch.linalg.vector_norm(A, ord=2, dim=1)
    print("Checking correctness of vector norm...")
    match = torch.allclose(norm_torch, norm_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_cross(device):
    """Test cross product along a given dimension between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if cross products match, 0 otherwise.
    """
    A = torch.rand((10, 3), dtype=torch.bfloat16, device=device)
    B = torch.rand((10, 3), dtype=torch.bfloat16, device=device)
    cross_nki = nki_linalg_cross(A, B, dim=1)
    cross_torch = torch.linalg.cross(A, B, dim=1)
    print("Checking correctness of cross product (linalg)...")
    match = torch.allclose(cross_torch, cross_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_outer(device):
    """Test outer product between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if outer products match, 0 otherwise.
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(12, dtype=torch.bfloat16, device=device)
    outer_nki = nki_linalg_outer(a, b)
    outer_torch = torch.outer(a, b)
    print("Checking correctness of outer product (linalg)...")
    match = torch.allclose(outer_torch, outer_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_tensordot(device):
    """Test tensordot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if tensordot results match, 0 otherwise.
    """
    A = torch.rand((4, 5, 6), dtype=torch.bfloat16, device=device)
    B = torch.rand((6, 7, 8), dtype=torch.bfloat16, device=device)
    tensordot_nki = nki_linalg_tensordot(A, B, dims=([2], [0]))
    tensordot_torch = torch.tensordot(A, B, dims=([2], [0]))
    print("Checking correctness of tensordot (linalg)...")
    match = torch.allclose(tensordot_torch, tensordot_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_eigh(device):
    """Test eigen decomposition (eigh) for symmetric matrices between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if eigenvalues match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    A = (A + A.transpose(-2, -1)) / 2  # make symmetric
    w_nki, v_nki = nki_linalg_eigh(A)
    w_torch, v_torch = torch.linalg.eigh(A)
    print("Checking correctness of eigh (eigen decomposition)...")
    match = torch.allclose(w_torch, w_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_eig(device):
    """Test eigen decomposition (eig) for square matrices between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if eigenvalues (real parts) match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    w_nki, v_nki = nki_linalg_eig(A)
    w_torch, v_torch = torch.linalg.eig(A)
    print("Checking correctness of eig (eigen decomposition)...")
    match = torch.allclose(w_torch.real, w_nki.real, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_slogdet(device):
    """Test sign and log-determinant computation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if sign and logdet match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device) + torch.eye(8, dtype=torch.bfloat16, device=device)*0.5
    sign_nki, logdet_nki = nki_linalg_slogdet(A)
    sign_torch, logdet_torch = torch.linalg.slogdet(A)
    print("Checking correctness of slogdet (sign and log-determinant)...")
    match = (torch.allclose(sign_torch, sign_nki, atol=1e-2, rtol=1e-2) and
             torch.allclose(logdet_torch, logdet_nki, atol=1e-2, rtol=1e-2))
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_solve(device):
    """Test linear system solver between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if solutions match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device) + torch.eye(8, dtype=torch.bfloat16, device=device)*0.5
    B = torch.rand((8, 3), dtype=torch.bfloat16, device=device)
    x_nki = nki_linalg_solve(A, B)
    x_torch = torch.linalg.solve(A, B)
    print("Checking correctness of linear system solve...")
    match = torch.allclose(x_torch, x_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_lstsq(device):
    """Test least-squares solver between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if solutions match, 0 otherwise.
    """
    A = torch.rand((10, 5), dtype=torch.bfloat16, device=device)
    B = torch.rand((10, 3), dtype=torch.bfloat16, device=device)
    sol_nki = nki_linalg_lstsq(A, B)
    sol_torch = torch.linalg.lstsq(A, B).solution
    print("Checking correctness of least-squares solve...")
    match = torch.allclose(sol_torch, sol_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_cholesky(device):
    """Test Cholesky decomposition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if Cholesky factors match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    A = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(8, dtype=torch.bfloat16, device=device)*0.1
    L_nki = nki_linalg_cholesky(A)
    L_torch = torch.linalg.cholesky(A)
    print("Checking correctness of Cholesky decomposition...")
    match = torch.allclose(L_torch, L_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_lu(device):
    """Test LU decomposition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if reconstructed matrices match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    P_nki, L_nki, U_nki = nki_linalg_lu(A)
    LU, pivots = torch.lu(A)
    P_torch, L_torch, U_torch = torch.lu_unpack(LU, pivots, A.shape)
    rec_nki = torch.matmul(P_nki, torch.matmul(L_nki, U_nki))
    rec_torch = torch.matmul(P_torch, torch.matmul(L_torch, U_torch))
    print("Checking correctness of LU decomposition (reconstruction)...")
    match = torch.allclose(rec_nki, rec_torch, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_ldl_factor(device):
    """Test LDL factorization between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if LDL factors match, 0 otherwise.
    """
    A = torch.rand((8, 8), dtype=torch.bfloat16, device=device)
    A = (A + A.transpose(-2, -1)) / 2  # make symmetric
    L_nki, D_nki = nki_linalg_ldl_factor(A)
    L_torch, D_torch = torch.linalg.ldl_factor(A)
    print("Checking correctness of LDL factorization...")
    match = torch.allclose(L_torch, L_nki, atol=1e-2, rtol=1e-2) and \
            torch.allclose(D_torch, D_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_triangular_solve(device):
    """Test triangular system solver between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on.
        
    Returns:
        int: Returns 1 if solutions match, 0 otherwise.
    """
    T = torch.tril(torch.rand((8, 8), dtype=torch.bfloat16, device=device))
    # Ensure T is non-singular by adding to the diagonal.
    T = T + torch.eye(8, dtype=torch.bfloat16, device=device)*0.5
    B = torch.rand((8, 3), dtype=torch.bfloat16, device=device)
    sol_nki = nki_linalg_triangular_solve(B, T, upper=False)
    sol_torch = torch.triangular_solve(B, T, upper=False).solution
    print("Checking correctness of triangular solve...")
    match = torch.allclose(sol_torch, sol_nki, atol=1e-2, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def main():
    device = xm.xla_device()
    test_torch_linalg_qr(device)
    test_torch_linalg_svd(device)
    test_torch_linalg_inv(device)
    test_torch_linalg_pinv(device)
    test_torch_linalg_matrix_norm(device)
    test_torch_linalg_vector_norm(device)
    test_torch_linalg_cross(device)
    test_torch_linalg_outer(device)
    test_torch_linalg_tensordot(device)
    test_torch_linalg_eigh(device)
    test_torch_linalg_eig(device)
    test_torch_linalg_slogdet(device)
    test_torch_linalg_solve(device)
    test_torch_linalg_lstsq(device)
    test_torch_linalg_cholesky(device)
    test_torch_linalg_lu(device)
    test_torch_linalg_ldl_factor(device)
    test_torch_linalg_triangular_solve(device)

if __name__ == "__main__":
    main()
