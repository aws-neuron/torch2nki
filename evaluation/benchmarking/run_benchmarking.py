import benchmarking_tools
import edgeops_tests, elementwise_tests, multi_element_tests, products_tests, matrixops_tests, mlops_tests


def main():
#Elementwise operators
    elementwise_operators = [
        "add", "sub", "mul", "div", "abs", "exp", "log", "sqrt", "rsqrt",
        "pow", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
        "tanh", "sigmoid", "relu", "threshold"
    ]

    #Multi-element operators
    multi_element_operators = [
        "softmax", "log_softmax", "max", "min", "sum", "mean", "var", "std", "norm",
        "cumsum", "cumprod", "prod", "round", "floor", "ceil", "trunc", "sign",
        "where", "eq", "ne", "gt", "lt", "clamp", "sort", "topk", "kthvalue", "median",
        "mode", "percentile", "logsumexp", "amax", "amin", "all", "any", "bincount",
        "unique", "unique_consecutive"
    ]

    #Products operators
    products_operators = [
        "inner", "outer", "dot", "vdot", "cross", "matmul", "mm", "mv", "bmm",
        "tensordot", "einsum", "kron", "hadamard", "linalg_vecdot", "linalg_multi_dot"
    ]

    #Decomposition 
    matrixops_operators  = [
        "linalg_qr", "linalg_svd", "linalg_inv", "linalg_pinv",
        "linalg_matrix_norm", "linalg_vector_norm", "linalg_cross",
        "linalg_outer", "linalg_tensordot", "linalg_eigh", "linalg_eig",
        "linalg_slogdet", "linalg_solve", "linalg_lstsq", "linalg_cholesky",
        "linalg_lu", "linalg_ldl_factor", "linalg_triangular_solve"
    ]

    #MlOps
    mlops_operators = [
        "gelu", "elu", "selu", "leaky_relu", "hardswish", "mse_loss", "l1_loss",
        "cross_entropy", "nll_loss", "binary_cross_entropy", "hinge_embedding_loss",
        "kl_div", "smooth_l1_loss", "cosine_embedding_loss", "triplet_margin_loss",
        "batch_norm", "layer_norm", "group_norm", "instance_norm", "dropout",
        "alpha_dropout", "feature_alpha_dropout", "softshrink", "euclidean_dist",
        "cosine_similarity", "pairwise_distance", "conv1d", "conv2d", "conv3d",
        "conv_transpose2d", "max_pool2d", "avg_pool2d"
    ]

    #EdgeOps
    edgeops_operators = [
        "special_entr", "special_i1", "special_xlogy", "special_logit",
        "angle", "polar", "view_as_real", "view_as_complex", "copysign",
        "nextafter", "hypot", "log1p", "expm1", "frexp", "ldexp",
        "logaddexp", "logaddexp2", "sinc", "xlogy", "edit_distance",
        "hamming_distance"
    ]

    #Create python files:
    benchmarking_tools.process(elementwise_operators, "elementwise_nki_kernels")
    benchmarking_tools.process(multi_element_operators, "multi_element_nki_kernels")
    benchmarking_tools.process(products_operators, "products_nki_kernels")
    benchmarking_tools.process(matrixops_operators, "matrixops_nki_kernels")
    benchmarking_tools.process(mlops_operators, "mlops_nki_kernels")
    benchmarking_tools.process(edgeops_operators, "edgeops_nki_kernels")



    #Run the benchmark tests
    edge_ops_dict = edgeops_tests.main()
    elementwise_ops_dict = elementwise_tests.main()
    multi_element_ops_dict = multi_element_tests.main()
    products_ops_dict = products_tests.main()
    matrixops_ops_dict = matrixops_tests.main()
    mlops_ops_dict = mlops_tests.main()

    #Save the results
    with open("results.json", "w") as f:
        json.dump({
            "edgeops": edge_ops_dict,
            "elementwise": elementwise_ops_dict,
            "multi_element": multi_element_ops_dict,
            "products": products_ops_dict,
            "matrixops": matrixops_ops_dict,
            "mlops": mlops_ops_dict
        }, f, indent=4)

if __name__ == "__main__":
    main()



