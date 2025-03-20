import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    size = v1.shape[0]
    
    result = nl.zeros(size, dtype=v1.dtype)

    for i in nl.arange(size):
        a = nl.load(v1[i:i+1])
        b = nl.load(v2[i:i+1])
        
        c = nl.add(a, b)

        nl.store(result[i:i+1], c)

    return result