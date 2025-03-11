import torch
import torch_xla
from torch_xla.core import xla_model as xm
import numpy as np

from edgeops_nki_kernels import (
    nki_special_entr, nki_special_i1, nki_special_xlogy, nki_special_logit,
    nki_angle, nki_polar, nki_view_as_real, nki_view_as_complex, nki_copysign,
    nki_nextafter, nki_hypot, nki_log1p, nki_expm1, nki_frexp, nki_ldexp,
    nki_logaddexp, nki_logaddexp2, nki_sinc, nki_xlogy,
    nki_edit_distance, nki_hamming_distance
)

def test_torch_special_entr(device):
    """Test special_entr (entropy function: x * log(x)) between NKI and reference implementation."""
    # Ensure positive values to avoid log(0)
    x = torch.rand((64, 128), dtype=torch.bfloat16, device=device) + 0.1
    out_nki = nki_special_entr(x)
    # Reference: x * log(x)
    out_ref = x * torch.log(x)
    print("Checking correctness of special_entr...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_special_i1(device):
    """Test special_i1 (modified Bessel function of the first kind, order 1) between NKI and reference implementation."""
    x = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    out_nki = nki_special_i1(x)
    # Use PyTorch's special function (convert to float32 then back to bfloat16)
    out_ref = torch.special.i1(x.float()).to(x.dtype)
    print("Checking correctness of special_i1...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_special_xlogy(device):
    """Test special_xlogy (computes x * log(y), even when x=0) between NKI and reference implementation."""
    # Create x with zeros and positive y
    x = torch.linspace(0, 1, steps=64, device=device, dtype=torch.bfloat16).unsqueeze(1).expand(64, 128)
    y = torch.rand((64, 128), dtype=torch.bfloat16, device=device) + 0.1
    out_nki = nki_special_xlogy(x, y)
    # Reference: when x==0, result is 0; otherwise x * log(y)
    out_ref = torch.where(x == 0, torch.zeros_like(x), x * torch.log(y))
    print("Checking correctness of special_xlogy...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_special_logit(device):
    """Test special_logit (inverse of sigmoid: logit function) between NKI and reference implementation."""
    # Input probabilities strictly in (0,1)
    x = torch.rand((64, 128), dtype=torch.bfloat16, device=device).clamp(0.01, 0.99)
    out_nki = nki_special_logit(x)
    out_ref = torch.logit(x.float()).to(x.dtype)
    print("Checking correctness of special_logit...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_angle(device):
    """Test angle (computes phase angle of a complex tensor) between NKI and reference implementation."""
    real = torch.randn((64, 128), device=device, dtype=torch.float32)
    imag = torch.randn((64, 128), device=device, dtype=torch.float32)
    x = torch.complex(real, imag)
    out_nki = nki_angle(x)
    out_ref = torch.angle(x)
    print("Checking correctness of angle...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_polar(device):
    """Test polar (converts magnitude and phase into a complex tensor) between NKI and reference implementation."""
    magnitude = torch.rand((64, 128), device=device, dtype=torch.float32)
    phase = torch.rand((64, 128), device=device, dtype=torch.float32) * 2 * np.pi - np.pi
    out_nki = nki_polar(magnitude, phase)
    out_ref = torch.polar(magnitude, phase)
    print("Checking correctness of polar...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_view_as_real(device):
    """Test view_as_real (converts a complex tensor into a real tensor with extra dimension) between NKI and reference implementation."""
    real = torch.randn((64, 128), device=device, dtype=torch.float32)
    imag = torch.randn((64, 128), device=device, dtype=torch.float32)
    x = torch.complex(real, imag)
    out_nki = nki_view_as_real(x)
    out_ref = torch.view_as_real(x)
    print("Checking correctness of view_as_real...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_view_as_complex(device):
    """Test view_as_complex (converts a real tensor with last dimension=2 into a complex tensor) between NKI and reference implementation."""
    x = torch.randn((64, 128, 2), device=device, dtype=torch.float32)
    out_nki = nki_view_as_complex(x)
    out_ref = torch.view_as_complex(x)
    print("Checking correctness of view_as_complex...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_copysign(device):
    """Test copysign (copies sign from one tensor to another) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device).abs()
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_nki = nki_copysign(x, y)
    out_ref = torch.copysign(x, y)
    print("Checking correctness of copysign...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_nextafter(device):
    """Test nextafter (finds next floating-point value after x in direction of y) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_nextafter(x, y)
    out_ref = torch.nextafter(x, y)
    print("Checking correctness of nextafter...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_hypot(device):
    """Test hypot (computes sqrt(x^2 + y^2)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_hypot(x, y)
    out_ref = torch.hypot(x, y)
    print("Checking correctness of hypot...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_log1p(device):
    """Test log1p (computes log(1 + x)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_log1p(x)
    out_ref = torch.log1p(x)
    print("Checking correctness of log1p...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_expm1(device):
    """Test expm1 (computes exp(x) - 1) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_expm1(x)
    out_ref = torch.expm1(x)
    print("Checking correctness of expm1...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_frexp(device):
    """Test frexp (returns mantissa and exponent) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    mantissa_nki, exponent_nki = nki_frexp(x)
    mantissa_ref, exponent_ref = torch.frexp(x)
    print("Checking correctness of frexp...")
    match = torch.allclose(mantissa_ref, mantissa_nki, atol=1e-5, rtol=1e-3) and torch.equal(exponent_ref, exponent_nki)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_ldexp(device):
    """Test ldexp (reconstructs float from mantissa and exponent) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    mantissa, exponent = torch.frexp(x)
    out_nki = nki_ldexp(mantissa, exponent)
    out_ref = torch.ldexp(mantissa, exponent)
    print("Checking correctness of ldexp...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_logaddexp(device):
    """Test logaddexp (computes log(exp(x) + exp(y))) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_logaddexp(x, y)
    out_ref = torch.logaddexp(x, y)
    print("Checking correctness of logaddexp...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_logaddexp2(device):
    """Test logaddexp2 (computes log2(2^x + 2^y)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_logaddexp2(x, y)
    out_ref = torch.logaddexp2(x, y)
    print("Checking correctness of logaddexp2...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_sinc(device):
    """Test sinc (computes sin(x)/x) between NKI and reference implementation."""
    x = torch.linspace(-10, 10, steps=128, device=device, dtype=torch.float32)
    out_nki = nki_sinc(x)
    out_ref = torch.sinc(x)
    print("Checking correctness of sinc...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_xlogy(device):
    """Test xlogy (computes x * log(y) handling x=0 correctly) between NKI and reference implementation."""
    x = torch.linspace(0, 1, steps=64, device=device, dtype=torch.float32).unsqueeze(1).expand(64, 128)
    y = torch.rand((64, 128), dtype=torch.float32, device=device) + 0.1
    out_nki = nki_xlogy(x, y)
    out_ref = torch.where(x == 0, torch.zeros_like(x), x * torch.log(y))
    print("Checking correctness of xlogy...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_edit_distance(device):
    """Test edit_distance (Levenshtein distance between two sequences) between NKI and a reference implementation."""
    seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
    seq2 = [1, 3, 4, 7, 8, 9]
    out_nki = nki_edit_distance(seq1, seq2)
    # Simple dynamic programming implementation for edit distance
    def edit_distance(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = min(dp[i-1][j] + 1,
                               dp[i][j-1] + 1,
                               dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1))
        return dp[m][n]
    out_ref = edit_distance(seq1, seq2)
    print("Checking correctness of edit_distance...")
    match = (out_nki == out_ref)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_hamming_distance(device):
    """Test hamming_distance (number of differing positions between two sequences) between NKI and a reference implementation."""
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 0, 3, 0, 5]
    out_nki = nki_hamming_distance(seq1, seq2)
    out_ref = sum(el1 != el2 for el1, el2 in zip(seq1, seq2))
    print("Checking correctness of hamming_distance...")
    match = (out_nki == out_ref)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def main():
    device = xm.xla_device()
    
    # Dictionary to store test results
    test_results = {}
    
    # Run all tests and store results
    test_results['special_entr'] = test_torch_special_entr(device)
    test_results['special_i1'] = test_torch_special_i1(device)
    test_results['special_xlogy'] = test_torch_special_xlogy(device)
    test_results['special_logit'] = test_torch_special_logit(device)
    test_results['angle'] = test_torch_angle(device)
    test_results['polar'] = test_torch_polar(device)
    test_results['view_as_real'] = test_torch_view_as_real(device)
    test_results['view_as_complex'] = test_torch_view_as_complex(device)
    test_results['copysign'] = test_torch_copysign(device)
    test_results['nextafter'] = test_torch_nextafter(device)
    test_results['hypot'] = test_torch_hypot(device)
    test_results['log1p'] = test_torch_log1p(device)
    test_results['expm1'] = test_torch_expm1(device)
    test_results['frexp'] = test_torch_frexp(device)
    test_results['ldexp'] = test_torch_ldexp(device)
    test_results['logaddexp'] = test_torch_logaddexp(device)
    test_results['logaddexp2'] = test_torch_logaddexp2(device)
    test_results['sinc'] = test_torch_sinc(device)
    test_results['xlogy'] = test_torch_xlogy(device)
    test_results['edit_distance'] = test_torch_edit_distance(device)
    test_results['hamming_distance'] = test_torch_hamming_distance(device)
    
    # Print summary of results
    print("\nTest Results Summary:")
    print("-" * 40)
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")
    
    # Print failed tests if any
    failed_tests = [test for test, result in test_results.items() if result == 0]
    if failed_tests:
        print("\nFailed Tests:")
        for test in failed_tests:
            print(f"- {test}")
    
    return test_results

if __name__ == "__main__":
    main()
