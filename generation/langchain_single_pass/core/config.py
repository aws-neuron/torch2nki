"""
Configuration management for the NKI kernel generation pipeline.
"""
import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for the kernel generation pipeline."""
    
    # Base paths - configurable
    base_path: str = "/home/ubuntu/torch2nki"
    
    # LLM Configuration
    llm_model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    llm_temperature: float = 0.85
    llm_max_tokens: int = 20000
    llm_top_p: float = 0.999
    llm_top_k: int = 250
    llm_region: str = "us-west-2"
    
    # Retry Configuration
    max_retries_per_operator: int = 30
    max_iterations_per_attempt: int = 15
    initial_backoff: float = 1.0
    
    # File paths
    @property
    def system_prompt_path(self) -> str:
        return os.path.join(self.base_path, "generation/langchain_single_pass/langchain_files/langchain_prompts/system_prompt_langchain.txt")
    
    @property
    def prompts_dir(self) -> str:
        return os.path.join(self.base_path, "prompts")
    
    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.base_path, "generation/langchain_single_pass/langchain_files/langchain_outputs")
    
    @property
    def error_doc_path(self) -> str:
        return os.path.join(self.base_path, "documentation/nki_documentation/nki_error_messages.txt")
    
    @property
    def docs_dir(self) -> str:
        return os.path.join(self.base_path, "documentation/nki_documentation/nki_language_apis_parsed")
    
    @property
    def tests_passed_dict_path(self) -> str:
        return os.path.join(self.outputs_dir, "test_passed_dict.json")


class OperatorConfig:
    """Configuration for different operator categories."""
    
    @staticmethod
    def get_elementwise_operators() -> List[str]:
        """Get list of elementwise operators."""
        return [
            "add", "sub", "mul", "div", "abs", "exp", "log", "sqrt", "rsqrt", 
            "pow", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", 
            "tanh", "sigmoid", "relu", "threshold"
        ]
    
    @staticmethod
    def get_elementwise_test_names() -> List[str]:
        """Get list of elementwise test function names."""
        return [
            "test_torch_addition", "test_torch_subtraction", "test_torch_multiplication",
            "test_torch_division", "test_torch_absolute", "test_torch_exponential",
            "test_torch_log", "test_torch_sqrt", "test_torch_rsqrt", "test_torch_power",
            "test_torch_sine", "test_torch_cosine", "test_torch_tangent",
            "test_torch_arcsine", "test_torch_arccosine", "test_torch_arctangent",
            "test_torch_hyperbolic_sine", "test_torch_hyperbolic_cosine",
            "test_torch_hyperbolic_tangent", "test_torch_sigmoid", "test_torch_relu",
            "test_torch_threshold"
        ]
    
    @staticmethod
    def get_multi_element_operators() -> List[str]:
        """Get list of multi-element operators."""
        return [
            "max", "min", "sum", "mean", "var", "std", "norm", "cumsum", "cumprod",
            "prod", "round", "floor", "ceil", "trunc", "sign", "where", "eq", "ne",
            "gt", "lt", "clamp", "sort", "topk", "kthvalue", "median", "mode",
            "percentile", "logsumexp", "amax", "amin", "all", "any", "bincount",
            "unique", "unique_consecutive"
        ]
    
    @staticmethod
    def get_multi_element_test_names() -> List[str]:
        """Get list of multi-element test function names."""
        return [
            "test_torch_max", "test_torch_min", "test_torch_sum", "test_torch_mean",
            "test_torch_var", "test_torch_std", "test_torch_norm", "test_torch_cumsum",
            "test_torch_cumprod", "test_torch_prod", "test_torch_round",
            "test_torch_floor", "test_torch_ceil", "test_torch_trunc",
            "test_torch_sign", "test_torch_where", "test_torch_eq", "test_torch_ne",
            "test_torch_gt", "test_torch_lt", "test_torch_clamp", "test_torch_sort",
            "test_torch_topk", "test_torch_kthvalue", "test_torch_median",
            "test_torch_mode", "test_torch_percentile", "test_torch_logsumexp",
            "test_torch_amax", "test_torch_amin", "test_torch_all", "test_torch_any",
            "test_torch_bincount", "test_torch_unique", "test_torch_unique_consecutive"
        ]
    
    @staticmethod
    def get_product_operators() -> List[str]:
        """Get list of product operators."""
        return ["sort"]  # Currently only testing sort
    
    @staticmethod
    def get_product_test_names() -> List[str]:
        """Get list of product test function names."""
        return ["test_torch_sort"]
    
    @staticmethod
    def get_operator_test_mapping() -> Dict[str, str]:
        """Get mapping from operator names to test function names."""
        mapping = {}
        
        # Elementwise operators
        elementwise_ops = OperatorConfig.get_elementwise_operators()
        elementwise_tests = OperatorConfig.get_elementwise_test_names()
        for op, test in zip(elementwise_ops, elementwise_tests):
            mapping[op] = test
        
        # Multi-element operators
        multi_ops = OperatorConfig.get_multi_element_operators()
        multi_tests = OperatorConfig.get_multi_element_test_names()
        for op, test in zip(multi_ops, multi_tests):
            mapping[op] = test
        
        # Product operators
        product_ops = OperatorConfig.get_product_operators()
        product_tests = OperatorConfig.get_product_test_names()
        for op, test in zip(product_ops, product_tests):
            mapping[op] = test
        
        return mapping


# Global configuration instance
config = PipelineConfig() 