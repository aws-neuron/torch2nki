import React, { useState } from "react";import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface Operator {
    name: string;
    description: string;
}

interface BenchmarkResult {
    edgeops: number;
    elementwise: number;
    multi_element: number;
    products: number;
    matrixops: number;
    mlops: number;
  }

const operatorGroups: { [key: string]: Operator[] } = {
  "Element Wise Operations": [
    { name: "add", description: "Element-wise addition" },
    { name: "sub", description: "Element-wise subtraction" },
    { name: "mul", description: "Element-wise multiplication" },
    { name: "div", description: "Element-wise division" },
    { name: "abs", description: "Element-wise absolute value" },
    { name: "exp", description: "Element-wise exponentiation" },
    { name: "log", description: "Element-wise natural logarithm" },
    { name: "sqrt", description: "Element-wise square root" },
    { name: "rsqrt", description: "Element-wise reciprocal square root" },
    { name: "pow", description: "Element-wise power function" },
    { name: "sin", description: "Element-wise sine" },
    { name: "cos", description: "Element-wise cosine" },
    { name: "tan", description: "Element-wise tangent" },
    { name: "asin", description: "Element-wise arcsine" },
    { name: "acos", description: "Element-wise arccosine" },
    { name: "atan", description: "Element-wise arctangent" },
    { name: "sinh", description: "Element-wise hyperbolic sine" },
    { name: "cosh", description: "Element-wise hyperbolic cosine" },
    { name: "tanh", description: "Element-wise hyperbolic tangent" },
    { name: "sigmoid", description: "Element-wise sigmoid activation" },
    { name: "relu", description: "Element-wise ReLU activation" },
    { name: "threshold", description: "Element-wise threshold operation" }
  ],
  "Multi-Element Vector Operations": [
    { name: "softmax", description: "Softmax function" },
    { name: "log_softmax", description: "Log softmax function" },
    { name: "max", description: "Element-wise maximum" },
    { name: "min", description: "Element-wise minimum" },
    { name: "sum", description: "Summation over tensor elements" },
    { name: "mean", description: "Mean over tensor elements" },
    { name: "var", description: "Variance computation" },
    { name: "std", description: "Standard deviation computation" },
    { name: "norm", description: "Norm computation" },
    { name: "cumsum", description: "Cumulative sum along a dimension" },
    { name: "cumprod", description: "Cumulative product along a dimension" },
    { name: "prod", description: "Product of tensor elements" },
    { name: "round", description: "Rounding to nearest integer" },
    { name: "floor", description: "Floor function" },
    { name: "ceil", description: "Ceil function" },
    { name: "trunc", description: "Truncate to integer" },
    { name: "sign", description: "Element-wise sign function" },
    { name: "where", description: "Element-wise conditional selection" },
    { name: "eq", description: "Element-wise equality comparison" },
    { name: "ne", description: "Element-wise inequality comparison" },
    { name: "gt", description: "Element-wise greater than comparison" },
    { name: "lt", description: "Element-wise less than comparison" },
    { name: "clamp", description: "Clamping values between min and max" }
  ],
  "Products": [
    { name: "inner", description: "Computes the inner product of two tensors." },
    { name: "outer", description: "Computes the outer product of two tensors." },
    { name: "dot", description: "Computes the dot product of two 1D tensors." },
    { name: "vdot", description: "Computes the dot product treating inputs as 1D." },
    { name: "cross", description: "Computes the cross product of two 3D vectors." },
    { name: "matmul", description: "Generalized matrix multiplication." },
    { name: "mm", description: "Matrix-matrix multiplication." },
    { name: "mv", description: "Matrix-vector multiplication." },
    { name: "bmm", description: "Batch matrix-matrix multiplication." },
    { name: "tensordot", description: "Computes tensor contraction over specified dimensions." },
    { name: "einsum", description: "Einstein summation notation for tensor operations." },
    { name: "kron", description: "Computes the Kronecker product of two tensors." },
    { name: "hadamard", description: "Computes element-wise (Hadamard) product of two tensors." },
    { name: "linalg_vecdot", description: "Computes the dot product of two vectors along a specified dimension." },
    { name: "linalg_multi_dot", description: "Computes the chain multiplication of multiple matrices." }
  ],
  "Decompositions & Matrix Operations": [
    { name: "linalg_qr", description: "Computes the QR decomposition of a matrix." },
    { name: "linalg_svd", description: "Computes the singular value decomposition of a matrix." },
    { name: "linalg_inv", description: "Computes the inverse of a square matrix." },
    { name: "linalg_pinv", description: "Computes the Moore-Penrose pseudo-inverse of a matrix." },
    { name: "linalg_matrix_norm", description: "Computes the norm of a matrix along specified dimensions." },
    { name: "linalg_vector_norm", description: "Computes the norm of a vector along a specified dimension." },
    { name: "linalg_cross", description: "Computes the cross product along a given dimension." },
    { name: "linalg_outer", description: "Computes the outer product of two vectors." },
    { name: "linalg_tensordot", description: "Computes tensor contractions along given dimensions." },
    { name: "linalg_eigh", description: "Computes the eigenvalues and eigenvectors of a symmetric matrix." },
    { name: "linalg_eig", description: "Computes the eigenvalues and eigenvectors of a square matrix." },
    { name: "linalg_slogdet", description: "Computes the sign and log-determinant of a square matrix." },
    { name: "linalg_solve", description: "Solves a system of linear equations Ax = B." },
    { name: "linalg_lstsq", description: "Solves a least-squares problem." },
    { name: "linalg_cholesky", description: "Computes the Cholesky decomposition of a positive definite matrix." },
    { name: "linalg_lu", description: "Computes the LU decomposition of a matrix." },
    { name: "linalg_ldl_factor", description: "Computes the LDL decomposition of a Hermitian matrix." },
    { name: "linalg_triangular_solve", description: "Solves a triangular system of equations." }
  ]
};


const Dashboard: React.FC = () => {
    const [results, setResults] = useState<{ [key: string]: BenchmarkResult | null }>({});
  
    const runBenchmark = async (operator: string): Promise<void> => {
      try {
        const response = await fetch("/api/run-benchmark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ operator })
        });
        const result: BenchmarkResult = await response.json();
        setResults((prev) => ({ ...prev, [operator]: result }));
      } catch (error) {
        console.error(`Error running benchmark for ${operator}:`, error);
      }
    };
  
    return (
      <div className="p-4">
        <h1 className="text-2xl font-bold mb-4">NKI Kernel Benchmarking</h1>
        {Object.entries(operatorGroups).map(([group, operators]) => (
          <div key={group} className="mb-6">
            <h2 className="text-xl font-semibold mb-2">{group}</h2>
            <div className="grid grid-cols-3 gap-4">
              {operators.map((operator) => (
                <Card key={operator.name} className="mb-2 p-2">
                  <CardContent>
                    <p className="text-lg font-semibold">{operator.name}</p>
                    <p className="text-sm text-gray-600 mb-2">{operator.description}</p>
                    <Button onClick={() => runBenchmark(operator.name)}>Run Benchmark</Button>
                    {results[operator.name] && (
                      <div className="mt-2 p-2 border rounded bg-gray-100 text-sm">
                        {Object.entries(results[operator.name] ?? {}).map(([key, value]) => (
                          <p key={key}><strong>{key}:</strong> {value}</p>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  export default Dashboard;