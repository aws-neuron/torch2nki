"""
Operator-specific execution logic for the NKI kernel generation pipeline.
"""
import json
from typing import Dict, List

from ..core.config import config, OperatorConfig
from ..pipeline.kernel_generator import kernel_generator


class OperatorRunner:
    """Handles operator-specific execution with retry logic."""
    
    def __init__(self):
        """Initialize the operator runner."""
        pass
    
    def run_operator_with_retries(self, operator: str, test_name: str) -> bool:
        """
        Run a single operator with retry logic.
        
        Args:
            operator: Name of the operator to generate
            test_name: Name of the test function to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        result = False
        ctr = 0
        
        while ctr < config.max_retries_per_operator:
            # Generate file paths for this attempt
            output_address = f"{config.outputs_dir}/{operator}_nki_kernel_attempt_{ctr}.txt"
            kernel_module_path = f"{config.outputs_dir}/{operator}_nki_kernel_attempt_{ctr}.py"
            test_script_output = f"{config.outputs_dir}/{operator}_error_message_attempt_{ctr}.txt"
            reasoning_log_path = f"{config.outputs_dir}/{operator}_reasoning_log_attempt_{ctr}.txt"
            
            # Generate user prompt path
            user_prompt_path = f"{config.prompts_dir}/{operator}_nki_prompt.txt"
            kernel_func_name = f"nki_{operator}"
            
            # Run the kernel generation
            result = kernel_generator.generate_kernel_with_direct_docs_and_error_loop(
                kernel_func_name,
                config.system_prompt_path,
                user_prompt_path,
                output_address,
                kernel_module_path,
                test_name,
                test_script_output,
                reasoning_log_path,
                config.error_doc_path,
                config.docs_dir,
                max_iterations=config.max_iterations_per_attempt
            )
            
            if result:
                print(f"Successfully generated kernel for {operator} on attempt {ctr + 1}")
                break
            else:
                print(f"Attempt {ctr + 1} failed for {operator}")
            
            ctr += 1
        
        return result
    
    def run_operators(self, operators: List[str], test_names: List[str]) -> Dict[str, bool]:
        """
        Run multiple operators and track their success.
        
        Args:
            operators: List of operator names
            test_names: List of corresponding test function names
            
        Returns:
            Dict mapping operator names to success status
        """
        tests_passed_dict = {}
        
        for i, operator in enumerate(operators):
            test_name = test_names[i]
            print(f"\n=== Processing operator: {operator} ===")
            
            result = self.run_operator_with_retries(operator, test_name)
            tests_passed_dict[operator] = result
            
            print(f"Result for {operator}: {'SUCCESS' if result else 'FAILED'}")
        
        return tests_passed_dict
    
    def run_all_operators(self) -> Dict[str, bool]:
        """Run all configured operators."""
        # Get all operator categories
        elementwise_operators = OperatorConfig.get_elementwise_operators()
        elementwise_test_names = OperatorConfig.get_elementwise_test_names()
        
        multi_element_operators = OperatorConfig.get_multi_element_operators()
        multi_element_test_names = OperatorConfig.get_multi_element_test_names()
        
        product_operators = OperatorConfig.get_product_operators()
        product_test_names = OperatorConfig.get_product_test_names()
        
        # Combine all operators and tests
        all_operators = elementwise_operators + multi_element_operators + product_operators
        all_test_names = elementwise_test_names + multi_element_test_names + product_test_names
        
        # Run all operators
        tests_passed_dict = self.run_operators(all_operators, all_test_names)
        
        # Save results to file
        with open(config.tests_passed_dict_path, "w") as f:
            json.dump(tests_passed_dict, f)
        
        return tests_passed_dict
    
    def run_specific_operators(self, operator_names: List[str]) -> Dict[str, bool]:
        """Run specific operators by name."""
        operator_test_mapping = OperatorConfig.get_operator_test_mapping()
        
        operators = []
        test_names = []
        
        for operator in operator_names:
            if operator in operator_test_mapping:
                operators.append(operator)
                test_names.append(operator_test_mapping[operator])
            else:
                print(f"Warning: Operator '{operator}' not found in configuration")
        
        if not operators:
            print("No valid operators found to run")
            return {}
        
        tests_passed_dict = self.run_operators(operators, test_names)
        
        # Save results to file
        with open(config.tests_passed_dict_path, "w") as f:
            json.dump(tests_passed_dict, f)
        
        return tests_passed_dict


# Global operator runner instance
operator_runner = OperatorRunner() 