import torch
import torch.nn as nn
import torch_xla
from torch_xla.core import xla_model as xm
import os
xla_device = xm.xla_device()  # or the appropriate method for your environment


import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import torch.nn.functional as F


def test_torch_addition(device, nki_vector_add):
    """Test elementwise addition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """

    # Test the small workload with basic kernel
    lhs_small = np.random.rand(300, 128).astype(np.float16)
    rhs_small = np.random.rand(300, 128).astype(np.float16)

    # Run NKI kernel
    output_small = torch.from_numpy(nki_vector_add(lhs_small, rhs_small))

    # Run torch reference
    output_small_torch = torch.add(torch.from_numpy(lhs_small), torch.from_numpy(rhs_small))

    # Compare results
    print("Checking correctness of nki_vector_add")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0


def test_torch_subtraction(device, nki_subtraction):
    """Test elementwise subtraction between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    lhs_small = np.random.rand(300, 128).astype(np.float16)
    rhs_small = np.random.rand(300, 128).astype(np.float16)

    # Run NKI kernel
    output_small = torch.from_numpy(nki_subtraction(lhs_small, rhs_small))

    # Run torch reference
    output_small_torch = torch.sub(torch.from_numpy(lhs_small), torch.from_numpy(rhs_small))

    # Compare results
    print("Checking correctness of nki_subtraction")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0


def test_torch_multiplication(device, nki_multiplication):
    """Test elementwise multiplication between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    lhs_small = np.random.rand(300, 128).astype(np.float16)
    rhs_small = np.random.rand(300, 128).astype(np.float16)

    # Run NKI kernel
    output_small = torch.from_numpy(nki_multiplication(lhs_small, rhs_small))

    # Run torch reference
    output_small_torch = torch.mul(torch.from_numpy(lhs_small), torch.from_numpy(rhs_small))

    # Compare results
    print("Checking correctness of nki_multiplication")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_division(device, nki_division):
    """Test elementwise division between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    lhs_small = np.random.rand(300, 128).astype(np.float16)
    rhs_small = np.random.rand(300, 128).astype(np.float16) + 0.1  # Avoid division by zero

    # Run NKI kernel
    output_small = torch.from_numpy(nki_division(lhs_small, rhs_small))

    # Run torch reference
    output_small_torch = torch.div(torch.from_numpy(lhs_small), torch.from_numpy(rhs_small))

    # Compare results
    print("Checking correctness of nki_division")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_absolute(device, nki_abs):
    """Test elementwise absolute value between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1

    # Run NKI kernel
    output_small = torch.from_numpy(nki_abs(x_small))

    # Run torch reference
    output_small_torch = torch.abs(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_abs")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_exponential(device, nki_exp):
    """Test elementwise exponential between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1

    # Run NKI kernel
    output_small = torch.from_numpy(nki_exp(x_small))

    # Run torch reference
    output_small_torch = torch.exp(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_exp")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_log(device, nki_log):
    """Test elementwise logarithm between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) + 0.1  # Avoid log(0)

    # Run NKI kernel
    output_small = torch.from_numpy(nki_log(x_small))

    # Run torch reference
    output_small_torch = torch.log(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_log")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sqrt(device, nki_sqrt):
    """Test elementwise square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16)  # Non-negative values

    # Run NKI kernel
    output_small = torch.from_numpy(nki_sqrt(x_small))

    # Run torch reference
    output_small_torch = torch.sqrt(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_sqrt")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_rsqrt(device, nki_rsqrt):
    """Test elementwise reciprocal square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) + 0.1  # Avoid division by zero

    # Run NKI kernel
    output_small = torch.from_numpy(nki_rsqrt(x_small))

    # Run torch reference
    output_small_torch = torch.rsqrt(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_rsqrt")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_power(device, nki_pow):
    """Test elementwise power between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) + 0.1  # Avoid negative values
    exponent = 2.0

    # Run NKI kernel
    output_small = torch.from_numpy(nki_pow(x_small, exponent))

    # Run torch reference
    output_small_torch = torch.pow(torch.from_numpy(x_small), exponent)

    # Compare results
    print("Checking correctness of nki_pow")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sine(device, nki_sin):
    """Test elementwise sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 * np.pi  # Values between 0 and 2π

    # Run NKI kernel
    output_small = torch.from_numpy(nki_sin(x_small))

    # Run torch reference
    output_small_torch = torch.sin(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_sin")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_cosine(device, nki_cos):
    """Test elementwise cosine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 * np.pi  # Values between 0 and 2π

    # Run NKI kernel
    output_small = torch.from_numpy(nki_cos(x_small))

    # Run torch reference
    output_small_torch = torch.cos(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_cos")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_ctc(device, nki_ctc):
    """Test the CTC loss kernel between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_ctc: The NKI CTC loss kernel to be tested
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Ensure input sizes are less than (128, 128)
    batch_size = np.random.randint(1, 128)  # Random batch size between 1 and 128
    max_input_length = np.random.randint(1, 128)  # Random input length between 1 and 128
    max_target_length = np.random.randint(1, 128)  # Random target length between 1 and 128
    num_classes = 6  # including the blank class

    # Generate random log probabilities and target sequences for testing
    log_probs = np.random.randn(max_input_length, batch_size, num_classes).astype(np.float32)
    targets = np.random.randint(1, num_classes, size=(batch_size, max_target_length)).astype(np.int32)
    input_lengths = np.array([max_input_length] * batch_size, dtype=np.int32)  # Lengths of input sequences
    target_lengths = np.array([max_target_length] * batch_size, dtype=np.int32)  # Lengths of target sequences

    # Convert to PyTorch tensors
    log_probs_tensor = torch.from_numpy(log_probs).to(device).log_softmax(dim=-1)  # Log probabilities
    targets_tensor = torch.from_numpy(targets).to(device)
    input_lengths_tensor = torch.from_numpy(input_lengths).to(device)
    target_lengths_tensor = torch.from_numpy(target_lengths).to(device)

    # Run NKI kernel (custom implementation)
    nki_ctc_loss = nki_ctc(log_probs_tensor, targets_tensor, input_lengths_tensor, target_lengths_tensor, blank=0)

    # Run PyTorch reference implementation
    torch_ctc_loss = F.ctc_loss(log_probs_tensor, targets_tensor, input_lengths_tensor, target_lengths_tensor, blank=0, reduction='sum')

    # Compare results
    print("Checking correctness of nki_ctc")
    print("NKI CTC loss:", nki_ctc_loss.item())
    print("Torch CTC loss:", torch_ctc_loss.item())

    # Compare the losses with a tolerance
    match = torch.isclose(nki_ctc_loss, torch_ctc_loss, atol=1e-4)
    print("NKI and Torch CTC loss match" if match else "Error: NKI and Torch CTC loss differ")
    
    return 1 if match else 0
    

def test_torch_tangent(device, nki_tan):
    """Test elementwise tangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 * np.pi  # Values between 0 and 2π

    # Run NKI kernel
    output_small = torch.from_numpy(nki_tan(x_small))

    # Run torch reference
    output_small_torch = torch.tan(torch.from_numpy(x_small))

    # Compare results
    print("Checking correctness of nki_tan")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_arcsine(device, nki_asin):
    """
    Test elementwise inverse sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_asin(x_small))
    
    # Run torch reference
    output_small_torch = torch.asin(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_asin")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_arccosine(device, nki_acos):
    """
    Test elementwise inverse cosine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_acos(x_small))
    
    # Run torch reference
    output_small_torch = torch.acos(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_acos")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_arctangent(device, nki_atan):
    """
    Test elementwise inverse tangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_atan(x_small))
    
    # Run torch reference
    output_small_torch = torch.atan(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_atan")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_hyperbolic_sine(device, nki_sinh):
    """
    Test elementwise hyperbolic sine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_sinh(x_small))
    
    # Run torch reference
    output_small_torch = torch.sinh(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_sinh")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_hyperbolic_cosine(device, nki_cosh):
    """
    Test elementwise hyperbolic cosine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_cosh(x_small))
    
    # Run torch reference
    output_small_torch = torch.cosh(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_cosh")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_hyperbolic_tangent(device, nki_tanh):
    """
    Test elementwise hyperbolic tangent between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_tanh(x_small))
    
    # Run torch reference
    output_small_torch = torch.tanh(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_tanh")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sigmoid(device, nki_sigmoid):
    """
    Test elementwise sigmoid between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_sigmoid(x_small))
    
    # Run torch reference
    output_small_torch = torch.sigmoid(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_sigmoid")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_relu(device, nki_relu):
    """
    Test elementwise ReLU between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_relu(x_small))
    
    # Run torch reference
    output_small_torch = torch.relu(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_relu")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_threshold(device, nki_threshold):
    """
    Test elementwise threshold between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1
    threshold = 0.5
    value = 0.0
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_threshold(x_small, threshold, value))
    
    # Run torch reference
    output_small_torch = torch.threshold(torch.from_numpy(x_small), threshold, value)
    
    print("Checking correctness of nki_threshold")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0


def test_torch_special_entr(device, nki_special_entr):
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

def test_torch_special_i1(device, nki_special_i1):
    """Test special_i1 (modified Bessel function of the first kind, order 1) between NKI and reference implementation."""
    x = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    out_nki = nki_special_i1(x)
    # Use PyTorch's special function (convert to float32 then back to bfloat16)
    out_ref = torch.special.i1(x.float()).to(x.dtype)
    print("Checking correctness of special_i1...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_special_xlogy(device, nki_special_xlogy):
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

def test_torch_special_logit(device, nki_special_logit):
    """Test special_logit (inverse of sigmoid: logit function) between NKI and reference implementation."""
    # Input probabilities strictly in (0,1)
    x = torch.rand((64, 128), dtype=torch.bfloat16, device=device).clamp(0.01, 0.99)
    out_nki = nki_special_logit(x)
    out_ref = torch.logit(x.float()).to(x.dtype)
    print("Checking correctness of special_logit...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_angle(device, nki_angle):
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

def test_torch_polar(device, nki_polar):
    """Test polar (converts magnitude and phase into a complex tensor) between NKI and reference implementation."""
    magnitude = torch.rand((64, 128), device=device, dtype=torch.float32)
    phase = torch.rand((64, 128), device=device, dtype=torch.float32) * 2 * np.pi - np.pi
    out_nki = nki_polar(magnitude, phase)
    out_ref = torch.polar(magnitude, phase)
    print("Checking correctness of polar...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_view_as_real(device, nki_view_as_real):
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

def test_torch_view_as_complex(device, nki_view_as_complex):
    """Test view_as_complex (converts a real tensor with last dimension=2 into a complex tensor) between NKI and reference implementation."""
    x = torch.randn((64, 128, 2), device=device, dtype=torch.float32)
    out_nki = nki_view_as_complex(x)
    out_ref = torch.view_as_complex(x)
    print("Checking correctness of view_as_complex...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_copysign(device, nki_copysign):
    """Test copysign (copies sign from one tensor to another) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device).abs()
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_nki = nki_copysign(x, y)
    out_ref = torch.copysign(x, y)
    print("Checking correctness of copysign...")
    match = torch.allclose(out_ref, out_nki, atol=1e-3, rtol=1e-2)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_nextafter(device, nki_nextafter):
    """Test nextafter (finds next floating-point value after x in direction of y) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_nextafter(x, y)
    out_ref = torch.nextafter(x, y)
    print("Checking correctness of nextafter...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_hypot(device, nki_hypot):
    """Test hypot (computes sqrt(x^2 + y^2)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_hypot(x, y)
    out_ref = torch.hypot(x, y)
    print("Checking correctness of hypot...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_log1p(device, nki_log1p):
    """Test log1p (computes log(1 + x)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_log1p(x)
    out_ref = torch.log1p(x)
    print("Checking correctness of log1p...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_expm1(device, nki_expm1):
    """Test expm1 (computes exp(x) - 1) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_expm1(x)
    out_ref = torch.expm1(x)
    print("Checking correctness of expm1...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_frexp(device, nki_frexp):
    """Test frexp (returns mantissa and exponent) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    mantissa_nki, exponent_nki = nki_frexp(x)
    mantissa_ref, exponent_ref = torch.frexp(x)
    print("Checking correctness of frexp...")
    match = torch.allclose(mantissa_ref, mantissa_nki, atol=1e-5, rtol=1e-3) and torch.equal(exponent_ref, exponent_nki)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_ldexp(device, nki_ldexp):
    """Test ldexp (reconstructs float from mantissa and exponent) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    mantissa, exponent = torch.frexp(x)
    out_nki = nki_ldexp(mantissa, exponent)
    out_ref = torch.ldexp(mantissa, exponent)
    print("Checking correctness of ldexp...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_logaddexp(device, nki_logaddexp):
    """Test logaddexp (computes log(exp(x) + exp(y))) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_logaddexp(x, y)
    out_ref = torch.logaddexp(x, y)
    print("Checking correctness of logaddexp...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_logaddexp2(device, nki_logaddexp2):
    """Test logaddexp2 (computes log2(2^x + 2^y)) between NKI and reference implementation."""
    x = torch.randn((64, 128), dtype=torch.float32, device=device)
    y = torch.randn((64, 128), dtype=torch.float32, device=device)
    out_nki = nki_logaddexp2(x, y)
    out_ref = torch.logaddexp2(x, y)
    print("Checking correctness of logaddexp2...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_sinc(device, nki_sinc):
    """Test sinc (computes sin(x)/x) between NKI and reference implementation."""
    x = torch.linspace(-10, 10, steps=128, device=device, dtype=torch.float32)
    out_nki = nki_sinc(x)
    out_ref = torch.sinc(x)
    print("Checking correctness of sinc...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_xlogy(device, nki_xlogy):
    """Test xlogy (computes x * log(y) handling x=0 correctly) between NKI and reference implementation."""
    x = torch.linspace(0, 1, steps=64, device=device, dtype=torch.float32).unsqueeze(1).expand(64, 128)
    y = torch.rand((64, 128), dtype=torch.float32, device=device) + 0.1
    out_nki = nki_xlogy(x, y)
    out_ref = torch.where(x == 0, torch.zeros_like(x), x * torch.log(y))
    print("Checking correctness of xlogy...")
    match = torch.allclose(out_ref, out_nki, atol=1e-5, rtol=1e-3)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0

def test_torch_edit_distance(device, nki_edit_distance):
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

def test_torch_hamming_distance(device, nki_hamming_distance):
    """Test hamming_distance (number of differing positions between two sequences) between NKI and a reference implementation."""
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 0, 3, 0, 5]
    out_nki = nki_hamming_distance(seq1, seq2)
    out_ref = sum(el1 != el2 for el1, el2 in zip(seq1, seq2))
    print("Checking correctness of hamming_distance...")
    match = (out_nki == out_ref)
    print("NKI and reference match!" if match else "NKI and reference differ")
    return 1 if match else 0



def test_torch_linalg_qr(device, nki_linalg_qr):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_svd(device, nki_linalg_svd):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_inv(device, nki_linalg_inv):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_pinv(device, nki_linalg_pinv):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_matrix_norm(device, nki_linalg_matrix_norm):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_vector_norm(device, nki_linalg_vector_norm):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_cross(device, nki_linalg_cross):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_outer(device, nki_linalg_outer):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_tensordot(device, nki_linalg_tensordot):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_eigh(device, nki_linalg_eigh):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_eig(device, nki_linalg_eig):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_slogdet(device, nki_linalg_slogdet):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_solve(device, nki_linalg_solve):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_lstsq(device, nki_linalg_lstsq):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_cholesky(device, nki_linalg_cholesky):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_lu(device, nki_linalg_lu):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_ldl_factor(device, nki_linalg_ldl_factor):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_triangular_solve(device, nki_linalg_triangular_solve):
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
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0



def test_torch_gelu(device, mlops_gelu):
    """Test GELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_gelu(x)
    out_torch = torch.nn.functional.gelu(x)
    print("Checking correctness of GELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_elu(device, mlops_elu):
    """Test ELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_elu(x, alpha=1.0)
    out_torch = torch.nn.functional.elu(x, alpha=1.0)
    print("Checking correctness of ELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_selu(device, mlops_selu):
    """Test SELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_selu(x)
    out_torch = torch.nn.functional.selu(x)
    print("Checking correctness of SELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_leaky_relu(device, mlops_leaky_relu):
    """Test Leaky ReLU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_leaky_relu(x, negative_slope=0.01)
    out_torch = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    print("Checking correctness of Leaky ReLU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_hardswish(device, mlops_hardswish):
    """Test Hard Swish activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_hardswish(x)
    out_torch = torch.nn.functional.hardswish(x)
    print("Checking correctness of Hard Swish activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_mse_loss(device, mlops_mse_loss):
    """Test MSE loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_mse_loss(input, target)
    loss_torch = torch.nn.functional.mse_loss(input, target)
    print("Checking correctness of MSE loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_l1_loss(device, mlops_l1_loss):
    """Test L1 loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_l1_loss(input, target)
    loss_torch = torch.nn.functional.l1_loss(input, target)
    print("Checking correctness of L1 loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_cross_entropy(device, mlops_cross_entropy):
    """Test cross entropy loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 10), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    loss_mlops = mlops_cross_entropy(input, target)
    loss_torch = torch.nn.functional.cross_entropy(input, target)
    print("Checking correctness of cross entropy loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_nll_loss(device, mlops_nll_loss):
    """Test NLL loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 10), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    loss_mlops = mlops_nll_loss(input, target)
    loss_torch = torch.nn.functional.nll_loss(input, target)
    print("Checking correctness of NLL loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_binary_cross_entropy(device, mlops_binary_cross_entropy):
    """Test binary cross entropy loss between MLOps and PyTorch implementations."""
    input = torch.sigmoid(torch.randn((64, 128), dtype=torch.bfloat16, device=device))
    target = torch.randint(0, 2, (64, 128), device=device).to(torch.bfloat16)
    loss_mlops = mlops_binary_cross_entropy(input, target)
    loss_torch = torch.nn.functional.binary_cross_entropy(input, target)
    print("Checking correctness of binary cross entropy loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_hinge_embedding_loss(device, mlops_hinge_embedding_loss):
    """Test hinge embedding loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 2, (64, 128), device=device) * 2 - 1
    loss_mlops = mlops_hinge_embedding_loss(input, target, margin=1.0)
    loss_torch = torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0)
    print("Checking correctness of hinge embedding loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_kl_div(device, mlops_kl_div):
    """Test KL divergence loss between MLOps and PyTorch implementations."""
    input = torch.log_softmax(torch.randn((64, 10), dtype=torch.bfloat16, device=device), dim=1)
    target = torch.softmax(torch.randn((64, 10), dtype=torch.bfloat16, device=device), dim=1)
    loss_mlops = mlops_kl_div(input, target, log_target=False)
    loss_torch = torch.nn.functional.kl_div(input, target, log_target=False)
    print("Checking correctness of KL divergence loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_smooth_l1_loss(device, mlops_smooth_l1_loss):
    """Test Smooth L1 loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_smooth_l1_loss(input, target)
    loss_torch = torch.nn.functional.smooth_l1_loss(input, target)
    print("Checking correctness of Smooth L1 loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_cosine_embedding_loss(device, mlops_cosine_embedding_loss):
    """Test cosine embedding loss between MLOps and PyTorch implementations."""
    input1 = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    input2 = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 2, (64,), device=device) * 2 - 1
    loss_mlops = mlops_cosine_embedding_loss(input1, input2, target)
    loss_torch = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
    print("Checking correctness of cosine embedding loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_triplet_margin_loss(device, mlops_triplet_margin_loss):
    """Test triplet margin loss between MLOps and PyTorch implementations."""
    anchor = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    positive = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    negative = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_triplet_margin_loss(anchor, positive, negative)
    loss_torch = torch.nn.functional.triplet_margin_loss(anchor, positive, negative)
    print("Checking correctness of triplet margin loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_batch_norm(device, mlops_batch_norm):
    """Test batch normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn(64, dtype=torch.bfloat16, device=device)
    bias = torch.randn(64, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_batch_norm(x, weight, bias, training=True)
    out_torch = torch.nn.functional.batch_norm(x, None, None, weight, bias, training=True)
    print("Checking correctness of batch normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_layer_norm(device, mlops_layer_norm):
    """Test layer normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32), dtype=torch.bfloat16, device=device)
    normalized_shape = (64, 32)
    weight = torch.randn(normalized_shape, dtype=torch.bfloat16, device=device)
    bias = torch.randn(normalized_shape, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_layer_norm(x, normalized_shape, weight, bias)
    out_torch = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias)
    print("Checking correctness of layer normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_group_norm(device, mlops_group_norm):
    """Test group normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn(64, dtype=torch.bfloat16, device=device)
    bias = torch.randn(64, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_group_norm(x, num_groups=8, weight=weight, bias=bias)
    out_torch = torch.nn.functional.group_norm(x, num_groups=8, weight=weight, bias=bias)
    print("Checking correctness of group normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_instance_norm(device, mlops_instance_norm):
    """Test instance normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_instance_norm(x, training=True)
    out_torch = torch.nn.functional.instance_norm(x, training=True)
    print("Checking correctness of instance normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_dropout(device, mlops_dropout):
    """Test dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.dropout(x, p=0.5, training=True)
    print("Checking correctness of dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_alpha_dropout(device, mlops_alpha_dropout):
    """Test alpha dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_alpha_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.alpha_dropout(x, p=0.5, training=True)
    print("Checking correctness of alpha dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_feature_alpha_dropout(device, mlops_feature_alpha_dropout):
    """Test feature alpha dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_feature_alpha_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.feature_alpha_dropout(x, p=0.5, training=True)
    print("Checking correctness of feature alpha dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_softshrink(device, mlops_softshrink):
    """Test softshrink activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_softshrink(x, lambd=0.5)
    out_torch = torch.nn.functional.softshrink(x, lambd=0.5)
    print("Checking correctness of softshrink activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_euclidean_dist(device, mlops_euclidean_dist, torch_norm):
    """Test Euclidean distance computation between MLOps and a reference implementation."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    dist_mlops = mlops_euclidean_dist(x, y)
    dist_ref = torch_norm(x - y, dim=1)
    print("Checking correctness of Euclidean distance computation...")
    match = torch.allclose(dist_ref, dist_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and reference match!" if match else "MLOps and reference differ")
    return 1 if match else 0

def test_torch_cosine_similarity(device, mlops_cosine_similarity, torch_cosine_similarity):
    """Test cosine similarity computation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    sim_mlops = mlops_cosine_similarity(x, y, dim=1)
    sim_torch = torch_cosine_similarity(x, y, dim=1)
    print("Checking correctness of cosine similarity computation...")
    match = torch.allclose(sim_torch, sim_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_pairwise_distance(device, mlops_pairwise_distance, torch_pairwise_distance):
    """Test pairwise distance computation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    dist_mlops = mlops_pairwise_distance(x, y)
    dist_torch = torch_pairwise_distance(x, y)
    print("Checking correctness of pairwise distance computation...")
    match = torch.allclose(dist_torch, dist_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv1d(device, mlops_conv1d):
    """Test 1D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 50), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv1d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv1d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv2d(device, mlops_conv2d):
    """Test 2D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 5, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv2d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv3d(device, mlops_conv3d):
    """Test 3D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((4, 3, 16, 16, 16), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 3, 3, 3), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv3d(x, weight, bias=None, stride=1, padding=1)
    out_torch = torch.nn.functional.conv3d(x, weight, bias=None, stride=1, padding=1)
    print("Checking correctness of conv3d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv_transpose2d(device, mlops_conv_transpose2d):
    """Test transposed 2D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 6, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn((3, 6, 5, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv_transpose2d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv_transpose2d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv_transpose2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_max_pool2d(device, mlops_max_pool2d):
    """Test 2D max pooling between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_max_pool2d(x, kernel_size=2, stride=2)
    out_torch = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
    print("Checking correctness of max_pool2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_avg_pool2d(device, mlops_avg_pool2d):
    """Test 2D average pooling between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_avg_pool2d(x, kernel_size=2, stride=2)
    out_torch = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    print("Checking correctness of avg_pool2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0



def test_torch_softmax(device, nki_softmax):
    """Test softmax operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """ 
    # Test with a small workload
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_softmax(x_small.to(xla_device))
    
    # Run torch reference
    output_small_torch = torch.softmax(x_small, dim=-1)
    
    # Compare results
    print("Checking correctness of softmax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_log_softmax(device, nki_log_softmax):
    """Test log softmax operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_log_softmax(x_small)
    output_small_torch = torch.log_softmax(x_small, dim=-1)
    print("Checking correctness of log softmax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_max(device, nki_max):
    """Test element-wise maximum operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_max(x_small, y_small)
    output_small_torch = torch.max(x_small, y_small)
    print("Checking correctness of element-wise maximum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_min(device, nki_min):
    """Test element-wise minimum operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_min(x_small, y_small)
    output_small_torch = torch.min(x_small, y_small)
    print("Checking correctness of element-wise minimum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sum(device, nki_sum):
    """Test summation operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_sum(x_small)
    output_small_torch = torch.sum(x_small)
    print("Checking correctness of summation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_mean(device, nki_mean):
    """Test mean operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_mean(x_small)
    output_small_torch = torch.mean(x_small)
    print("Checking correctness of mean operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_var(device, nki_var):
    """Test variance operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_var(x_small)
    output_small_torch = torch.var(x_small)
    print("Checking correctness of variance operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_std(device, nki_std):
    """Test standard deviation operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_std(x_small)
    output_small_torch = torch.std(x_small)
    print("Checking correctness of standard deviation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_norm(device, nki_norm):
    """Test norm operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_norm(x_small)
    output_small_torch = torch.norm(x_small)
    print("Checking correctness of norm operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_cumsum(device, nki_cumsum):
    """Test cumulative sum operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_cumsum(x_small, dim=-1)
    output_small_torch = torch.cumsum(x_small, dim=-1)
    print("Checking correctness of cumulative sum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_cumprod(device, nki_cumprod):
    """Test cumulative product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Add a small constant to avoid multiplying by zero
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) + 0.1
    output_small = nki_cumprod(x_small, dim=-1)
    output_small_torch = torch.cumprod(x_small, dim=-1)
    print("Checking correctness of cumulative product operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_prod(device, nki_prod):
    """Test product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) + 0.1
    output_small = nki_prod(x_small)
    output_small_torch = torch.prod(x_small)
    print("Checking correctness of product operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_round(device, nki_round):
    """Test rounding operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_round(x_small)
    output_small_torch = torch.round(x_small)
    print("Checking correctness of rounding operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_floor(device, nki_floor):
    """Test floor operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_floor(x_small)
    output_small_torch = torch.floor(x_small)
    print("Checking correctness of floor operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_ceil(device, nki_ceil):
    """Test ceil operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_ceil(x_small)
    output_small_torch = torch.ceil(x_small)
    print("Checking correctness of ceil operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_trunc(device, nki_trunc):
    """Test truncation operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_trunc: NKI trunc function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_trunc(x_small)
    output_small_torch = torch.trunc(x_small)
    print("Checking correctness of truncation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sign(device, nki_sign):
    """Test sign operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_sign: NKI sign function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randn((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_sign(x_small)
    output_small_torch = torch.sign(x_small)
    print("Checking correctness of sign operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_where(device, nki_where):
    """Test element-wise conditional selection (where) between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_where: NKI where function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    other = torch.zeros_like(x_small)
    output_small = nki_where(condition, x_small, other)
    output_small_torch = torch.where(condition, x_small, other)
    print("Checking correctness of element-wise conditional selection (where)...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_eq(device, nki_eq):
    """Test element-wise equality comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_eq: NKI equality comparison function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = x_small.clone()
    output_small = nki_eq(x_small, y_small)
    output_small_torch = torch.eq(x_small, y_small)
    print("Checking correctness of element-wise equality comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_ne(device, nki_ne):
    """Test element-wise inequality comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_ne: NKI inequality comparison function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = x_small + 1.0  # ensure differences
    output_small = nki_ne(x_small, y_small)
    output_small_torch = torch.ne(x_small, y_small)
    print("Checking correctness of element-wise inequality comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_gt(device, nki_gt):
    """Test element-wise greater than comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_gt: NKI greater than comparison function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_gt(x_small, y_small)
    output_small_torch = torch.gt(x_small, y_small)
    print("Checking correctness of element-wise greater than comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_lt(device, nki_lt):
    """Test element-wise less than comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_lt: NKI less than comparison function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_lt(x_small, y_small)
    output_small_torch = torch.lt(x_small, y_small)
    print("Checking correctness of element-wise less than comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_clamp(device, nki_clamp):
    """Test clamping operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_clamp: NKI clamp function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2  # values in [0,2]
    output_small = nki_clamp(x_small, min=0.5, max=1.5)
    output_small_torch = torch.clamp(x_small, min=0.5, max=1.5)
    print("Checking correctness of clamping operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_sort(device, nki_sort):
    """Test sort operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_sort: NKI sort function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    values_small, indices_small = nki_sort(x_small, dim=-1)
    output_small_torch = torch.sort(x_small, dim=-1)
    values_small_torch, indices_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of sort operation...")
    match = torch.allclose(values_small_torch, values_small, atol=1e-4, rtol=1e-2) and torch.equal(indices_small_torch, indices_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_topk(device, nki_topk):
    """Test top-k operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_topk: NKI top-k function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    k = 5
    values_small, indices_small = nki_topk(x_small, k=k, dim=-1)
    output_small_torch = torch.topk(x_small, k=k, dim=-1)
    values_small_torch, indices_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of top-k operation...")
    match = torch.allclose(values_small_torch, values_small, atol=1e-4, rtol=1e-2) and torch.equal(indices_small_torch, indices_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_kthvalue(device, nki_kthvalue):
    """Test kth value operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_kthvalue: NKI kth value function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    k = 10
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    value_small, index_small = nki_kthvalue(x_small, k=k, dim=-1)
    output_small_torch = torch.kthvalue(x_small, k=k, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of kth value operation...")
    match = torch.allclose(value_small_torch, value_small, atol=1e-4, rtol=1e-2) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_median(device, nki_median):
    """Test median operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_median: NKI median function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    value_small, index_small = nki_median(x_small, dim=-1)
    output_small_torch = torch.median(x_small, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of median operation...")
    match = torch.allclose(value_small_torch, value_small, atol=1e-4, rtol=1e-2) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_mode(device, nki_mode):
    """Test mode operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_mode: NKI mode function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Use an integer tensor with limited range to obtain meaningful mode
    x_small = torch.randint(0, 5, (300, 128), device=device)
    value_small, index_small = nki_mode(x_small, dim=-1)
    output_small_torch = torch.mode(x_small, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of mode operation...")
    match = torch.equal(value_small_torch, value_small) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_percentile(device, nki_percentile):
    """Test percentile operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_percentile: NKI percentile function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Use 50th percentile as a test (equivalent to median)
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_percentile(x_small, q=50, dim=-1)
    output_small_torch = torch.percentile(x_small, q=50, dim=-1)
    print("Checking correctness of percentile operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_logsumexp(device, nki_logsumexp):
    """Test logsumexp operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_logsumexp: NKI logsumexp function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_logsumexp(x_small, dim=-1)
    output_small_torch = torch.logsumexp(x_small, dim=-1)
    print("Checking correctness of logsumexp operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_amax(device, nki_amax):
    """Test amax operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_amax: NKI amax function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_amax(x_small, dim=-1)
    output_small_torch = torch.amax(x_small, dim=-1)
    print("Checking correctness of amax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_amin(device, nki_amin):
    """Test amin operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_amin: NKI amin function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_amin(x_small, dim=-1)
    output_small_torch = torch.amin(x_small, dim=-1)
    print("Checking correctness of amin operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_all(device, nki_all):
    """Test all operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_all: NKI all function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    output_small = nki_all(condition)
    output_small_torch = torch.all(condition)
    print("Checking correctness of all operation...")
    match = output_small_torch.item() == output_small.item()
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_any(device, nki_any):
    """Test any operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_any: NKI any function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    output_small = nki_any(condition)
    output_small_torch = torch.any(condition)
    print("Checking correctness of any operation...")
    match = output_small_torch.item() == output_small.item()
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_bincount(device, nki_bincount):
    """Test bincount operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_bincount: NKI bincount function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randint(0, 10, (100,), device=device)
    output_small = nki_bincount(x_small)
    output_small_torch = torch.bincount(x_small)
    print("Checking correctness of bincount operation...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_unique(device, nki_unique):
    """Test unique operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_unique: NKI unique function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randint(0, 10, (100,), device=device)
    output_small = nki_unique(x_small, return_counts=True)
    output_small_torch = torch.unique(x_small, return_counts=True)
    print("Checking correctness of unique operation...")
    match = torch.equal(output_small_torch[0], output_small[0]) and torch.equal(output_small_torch[1], output_small[1])
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0



def test_torch_unique_consecutive(device, nki_unique_consecutive):
    """Test unique consecutive operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        nki_unique_consecutive: NKI unique consecutive function
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.tensor([1, 1, 2, 2, 3, 3, 2, 2, 1, 1], device=device)
    output_small = nki_unique_consecutive(x_small)
    output_small_torch = torch.unique_consecutive(x_small)
    print("Checking correctness of unique consecutive operation...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_inner(device):
    """Test inner product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_inner(a, b)
    output_torch = torch.inner(a, b)
    print("Checking correctness of inner product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_outer(device):
    """Test outer product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(12, dtype=torch.bfloat16, device=device)
    output_nki = nki_outer(a, b)
    output_torch = torch.outer(a, b)
    print("Checking correctness of outer product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_dot(device):
    """Test dot product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_dot(a, b)
    output_torch = torch.dot(a, b)
    print("Checking correctness of dot product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_vdot(device):
    """Test vdot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_vdot(a, b)
    output_torch = torch.vdot(a, b)
    print("Checking correctness of vdot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_cross(device):
    """Test cross product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.randn(3, dtype=torch.bfloat16, device=device)
    b = torch.randn(3, dtype=torch.bfloat16, device=device)
    output_nki = nki_cross(a, b)
    output_torch = torch.cross(a, b)
    print("Checking correctness of cross product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_matmul(device):
    """Test matrix multiplication (matmul) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_matmul(a, b)
    output_torch = torch.matmul(a, b)
    print("Checking correctness of matmul operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_mm(device):
    """Test matrix-matrix multiplication (mm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_mm(a, b)
    output_torch = torch.mm(a, b)
    print("Checking correctness of mm operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_mv(device):
    """Test matrix-vector multiplication (mv) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128,), dtype=torch.bfloat16, device=device)
    output_nki = nki_mv(a, b)
    output_torch = torch.mv(a, b)
    print("Checking correctness of mv operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_bmm(device):
    """Test batch matrix-matrix multiplication (bmm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((10, 300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((10, 128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_bmm(a, b)
    output_torch = torch.bmm(a, b)
    print("Checking correctness of bmm operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_tensordot(device):
    """Test tensordot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((4, 5, 6), dtype=torch.bfloat16, device=device)
    b = torch.rand((6, 7, 8), dtype=torch.bfloat16, device=device)
    output_nki = nki_tensordot(a, b, dims=([2], [0]))
    output_torch = torch.tensordot(a, b, dims=([2], [0]))
    print("Checking correctness of tensordot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_einsum(device):
    """Test einsum operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    equation = "ij,jk->ik"
    output_nki = nki_einsum(equation, a, b)
    output_torch = torch.einsum(equation, a, b)
    print("Checking correctness of einsum operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_kron(device):
    """Test Kronecker product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    b = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    output_nki = nki_kron(a, b)
    output_torch = torch.kron(a, b)
    print("Checking correctness of Kronecker product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_hadamard(device):
    """Test Hadamard (element-wise multiplication) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((300, 128), dtype=torch.bfloat16, device=device)
    output_nki = nki_hadamard(a, b)
    output_torch = torch.mul(a, b)
    print("Checking correctness of Hadamard product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_vecdot(device):
    """Test linalg_vecdot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    b = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    output_nki = nki_linalg_vecdot(a, b, dim=1)
    output_torch = torch.linalg.vecdot(a, b, dim=1)
    print("Checking correctness of linalg_vecdot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_multi_dot(device):
    """Test linalg_multi_dot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    A = torch.rand((10, 20), dtype=torch.bfloat16, device=device)
    B = torch.rand((20, 30), dtype=torch.bfloat16, device=device)
    C = torch.rand((30, 40), dtype=torch.bfloat16, device=device)
    matrices = [A, B, C]
    output_nki = nki_linalg_multi_dot(matrices)
    output_torch = torch.linalg.multi_dot(matrices)
    print("Checking correctness of linalg_multi_dot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_relu(device, nki_relu):
    """
    Test elementwise ReLU between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    x_small = np.random.rand(300, 128).astype(np.float16) * 2 - 1
    
    # Run NKI kernel
    output_small = torch.from_numpy(nki_relu(x_small))
    
    # Run torch reference
    output_small_torch = torch.relu(torch.from_numpy(x_small))
    
    print("Checking correctness of nki_relu")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

def test_torch_threshold(device, nki_threshold):
    """
    Test elementwise threshold between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((300, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    threshold = 0.5
    value = 0.0
    
    # Run NKI kernel
    output_small = nki_threshold(x_small, threshold, value)
    
    # Run torch reference
    output_small_torch = torch.threshold(x_small, threshold, value)
    
    print("Checking correctness of nki_threshold")
    print("NKI output:", output_small)
    print("Torch output:", output_small_torch)
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "Error: NKI and Torch differ")
    return 1 if match else 0

