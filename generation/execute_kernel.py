import os
import re
import subprocess
from openai import OpenAI

######################
# OpenAI Kernel Generation Functions
######################

def get_chat_completion(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.7):
    """
    Returns the completion for the given system prompt and user prompt.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content

def run_o4(system_prompt_address, user_prompt_address, output_address):
    """
    Reads the system and user prompt files, calls the OpenAI API to generate a completion,
    and saves the raw output to output_address.
    """
    with open(system_prompt_address, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(user_prompt_address, "r", encoding="utf-8") as f:
        user_prompt = f.read()
    
    result = get_chat_completion(system_prompt, user_prompt)
    
    with open(output_address, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"OpenAI output saved to {output_address}")

######################
# Extraction Functions
######################

def extract_kernel_from_llm_response(file_path):
    """
    Reads the LLM-generated file, locates the Python code block (enclosed by triple backticks),
    and extracts only the code inside.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
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

######################
# Script Execution Function
######################

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

######################
# File Utility Functions
######################

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

######################
# Main Workflow Loop
######################

if __name__ == '__main__':
    # Define constant file paths (update these as needed)
    system_prompt_address = "/home/ubuntu/torch2nki/prompts/system_prompt_naive.txt"
    # Start with an initial user prompt file; subsequent iterations will update this.
    user_prompt_address   = "/home/ubuntu/torch2nki/prompts/user_prompt_add.txt"
    openai_output_address = "/home/ubuntu/torch2nki/generation/samples/vector_add.txt"   # Raw OpenAI output
    kernel_module_path    = "/home/ubuntu/torch2nki/generation/samples/vector_add_kernel.py"  # Kernel module file
    test_script_path      = "/home/ubuntu/torch2nki/evaluation/samples/test_vector_add.py"
    test_script_output    = "/home/ubuntu/torch2nki/prompts/script_output.txt"
    # Base prompt file: additional instructions to be appended (optional)
    base_prompt_path      = "/home/ubuntu/torch2nki/prompts/user_prompt_add.txt"
    # File where the reasoning from each iteration will be logged.
    reasoning_log_path    = "/home/ubuntu/torch2nki/generation/samples/reasoning_log.txt"
    
    # We'll write the new user prompt into this file and then update user_prompt_address accordingly.
    new_user_prompt_path  = "/home/ubuntu/torch2nki/prompts/new_user_prompt.txt"

    max_iterations = 5  # Set the number of iterations you want

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # 1. Generate kernel code via OpenAI API call
        run_o4(system_prompt_address, user_prompt_address, openai_output_address)
        
        # 1.1. Extract and log the reasoning from the raw OpenAI output.
        openai_output_text = read_file(openai_output_address)
        reasoning_text = extract_reasoning(openai_output_text)
        if reasoning_text:
            with open(reasoning_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"=== Iteration {iteration + 1} ===\n")
                log_file.write(reasoning_text)
                log_file.write("\n\n")
            print("Reasoning extracted and appended to reasoning log.")
        else:
            print("No reasoning found in the output.")
        
        # 2. Extract the Python kernel code from the generated output.
        try:
            kernel_code = extract_kernel_from_llm_response(openai_output_address)
        except ValueError as e:
            print(f"Error: {e}")
            break

        # 3. Write the extracted kernel code to a Python module file.
        write_file(kernel_module_path, kernel_code)
        print(f"Kernel code saved to {kernel_module_path}")

        # 4. Run the simulation/test script and capture its output.
        run_script_and_save_output(test_script_path, test_script_output)

        # 5. Read the current kernel code and error message.
        kernel_code_current = read_file(kernel_module_path)
        error_message = read_file(test_script_output)

        # 6. Build a new user prompt that includes a header with the kernel and error message.
        header = (
            "Here is the kernel you just wrote:\n"
            "--------------------------------------------------\n"
            f"{kernel_code_current}\n"
            "--------------------------------------------------\n\n"
            "Here is the error message it got:\n"
            "--------------------------------------------------\n"
            f"{error_message}\n"
            "--------------------------------------------------\n\n"
            "Try to fix it. Clearly explain your line of reasoning as well as what you think the error is, and how you plan to fix it. " \
            "Clearly put your initial reasoning inside triple stars like this *** example: i am making this change because i love unicorns ***. " \
            "I want all your initial reasoning inside of these triple stars, not just the summary at the end."
        )

        # Append additional base prompt instructions if available.
        if os.path.exists(base_prompt_path):
            base_prompt = read_file(base_prompt_path)
        else:
            base_prompt = ""

        new_user_prompt = header + "\n\n" + base_prompt

        # Write the new prompt to a file.
        write_file(new_user_prompt_path, new_user_prompt)
        print(f"New user prompt saved to {new_user_prompt_path}")

        # Update the user prompt address for the next iteration.
        user_prompt_address = new_user_prompt_path

        # Pause for review before the next iteration.
        input("Press Enter to continue to the next iteration (or Ctrl+C to exit)...")
