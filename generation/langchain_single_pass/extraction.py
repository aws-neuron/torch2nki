import subprocess
import datetime
import re

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
