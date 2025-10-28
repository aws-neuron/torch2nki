import subprocess
import datetime
import re

def update_function_name_in_text(text, new_name):
    """
    Updates the function name in the function header of a text string.

    The function expects the function header to follow this format:
    def old_function_name(arguments):
        <body lines>

    Args:
        text (str): The text content to update
        new_name (str): New function name to replace the old one with

    Returns:
        str: The updated text content with the new function name
    """
    # Updated regex to capture standard Python function definitions
    pattern = r'^(def\s+)([^\s(]+)(\s*\(.*\):)'  # Matches 'def function_name(args):'
    # Replace with new function name while preserving 'def' and arguments
    replacement = r'\1' + new_name + r'\3'
    # Replace the first occurrence of the function definition
    new_text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    
    return new_text


def extract_kernel_from_llm_response(content):
    """
    Locates the Python code block (enclosed by triple backticks) in the content,
    and extracts only the code inside.
    """
    pattern = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL)
    match = pattern.search(content)
    if not match:
        raise ValueError("Could not find a fenced Python code block in the generated output.")
    
    kernel_code = match.group(1)
    return kernel_code.strip()

def extract_reasoning(completion_text):
    """
    Extracts any text enclosed in triple stars (*** ... ***) from the completion text.
    Returns a string with all found reasoning (each block separated by a newline).
    """
    pattern = re.compile(r"\*\*\*\s*(.*?)\s*\*\*\*", re.DOTALL)
    matches = pattern.findall(completion_text)
    if matches:
        return "\n".join(matches)
    else:
        return ""

def run_script_and_save_output(script_path, output_file):
    """
    Executes a Python script and captures its stdout and stderr.
    """
    result = subprocess.run(
        ['python', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    combined_output = result.stdout + "\n" + result.stderr
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_output)
    
    print(f"Test script output saved to {output_file}")
    return combined_output

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def log_to_file(log_file_path, message, append=True):
    """Log a message to a file, with option to append or overwrite."""
    mode = "a" if append else "w"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, mode, encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

class ExecutionServer:
    """A server capable of running test functions with specified device and NKI function."""
    
    def __init__(self, device='cpu'):
        """Initialize the execution server.
        
        Args:
            device: The device to run tests on (default: 'cpu')
        """
        self.device = device
        import tests
        self.tests = tests
    
    @staticmethod
    def load_kernel_module(kernel_path):
        """Dynamically load the kernel module from the given path."""
        import importlib.util
        import os
        import sys
        
        # Remove .py extension if present
        if kernel_path.endswith('.py'):
            kernel_path = kernel_path[:-3]
            
        # Get module name from path
        module_name = os.path.basename(kernel_path)
        
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, kernel_path + '.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def run(self, test_func_name, kernel_func_name, kernel_module_path, output_file):
        """Run a test function with the specified NKI function and save output.
        
        Args:
            test_func_name: The name of the test function from tests.py to run
            kernel_module_path: Path to the kernel module to test
            output_file: Path to save the output to
            
        Returns:
            The combined stdout and stderr output from running the test
        """
        import sys
        from io import StringIO
        
        # Load the kernel module
        try:
            kernel_module = self.load_kernel_module(kernel_module_path)
        except Exception as e:
            error = f"Error loading kernel module: {str(e)}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(error)
            return error
        
        # Capture stdout and stderr
        stdout = StringIO()
        stderr = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        
        try:
            test_func = getattr(self.tests, test_func_name)
            # Get the kernel function - it should have the same name as the operator
            kernel_func = getattr(kernel_module, kernel_func_name)
            test_func(self.device, kernel_func)
        except Exception as e:
            print(f"Error running test: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore stdout and stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
        # Get the output
        output = stdout.getvalue() + "\n" + stderr.getvalue()
        stdout.close()
        stderr.close()
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        
        print(f"Test output saved to {output_file}")
        return output

def run(test_func_name, kernel_func_name, kernel_module_path, output_file, device='cpu'):
    """Run a test function using an execution server and save output.
    
    Args:
        test_func_name: The name of the test function from tests.py to run (e.g., 'test_torch_addition')
        kernel_func_name: The name of the kernel function to test (e.g., 'nki_vector_add')
        kernel_module_path: Path to the kernel module to test
        output_file: Path to save the output to
        device: The device to run on (default: 'cpu')
        
    Returns:
        The combined stdout and stderr output from running the test
    """
    server = ExecutionServer(device)
    return server.run(test_func_name, kernel_func_name, kernel_module_path, output_file)
