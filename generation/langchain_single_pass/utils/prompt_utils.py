"""
Prompt utility functions for the NKI kernel generation pipeline.
"""
import re
from typing import List, Optional


def extract_kernel_from_llm_response(content: str) -> str:
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


def extract_reasoning(completion_text: str) -> str:
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


def update_function_name_in_text(text: str, new_name: str) -> str:
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


def create_initial_generation_prompt(system_prompt: str, user_prompt: str, function_docs: str) -> str:
    """Create the initial kernel generation prompt."""
    return (
        f"{system_prompt}\n\n"
        f"Task: {user_prompt}\n\n"
        f"Function Documentation:\n{function_docs}\n\n"
        f"Generate a NKI kernel for the task."
    )


def create_enhanced_error_prompt_template() -> str:
    """Create the enhanced error re-injection prompt template."""
    return (
        "{system_prompt}\n\n"
        "Generate a new improved kernel for this task. Clearly explain your line of reasoning in one sentence, trying"
        "to keep it as brief as possible. Focus on explaining the exact change you will be making to the code."
        "I dont want the actual code, but be specific so someone that sees the same error message on a different line of code"
        "can implement the same fix. Remember to keep it concise, but explanatory as you will be referencing this later to make sure"
        "you are not trying to do the same fixes multiple times. "
        "When you are changing the code, try to only change the line with the error message and maybe code that relates."
        "However, if the error you are facing is that the outputs differ, then you are allowed to change multiple lines."
        "When the outputs differ, most likely the logic is wrong. I want you to notice this and in your reasoning state that the logic is "
        "likely wrong and state which logic you will update. Please clearly state in your reasoning ***i see that the outputs differ***"
        "Your output should include the entire kernel code, NOT just individual fixes. I want to be able to run the code inside the ``` ```"
        "The way I want your response structured is an explanation of your reasoning at the very start inside *** *** triple stars. "
        "Then, immediatly after write the python nki code inside triple backticks ``` ```."
        "I repeat, I only want your output to first be the line of reasoning inside triple stars, then the "
        "nki kernel code inside triple backticks. Do NOT put the reasoning inside the nki kernel code."
        "Everything above this line is the most important information. Please make sure you follow these guidelines."
        "Task: {user_prompt}\n\n"
        
        "{iteration_history}\n\n"
        "Previous error message:\n"
        "--------------------------------------------------\n"
        "{previous_error_message}\n"
        "--------------------------------------------------\n\n"
        "Function Documentation:\n"
        "--------------------------------------------------\n"
        "{function_docs}\n"
        "--------------------------------------------------\n\n"
    )


def create_error_selection_prompt(error_message: str, available_errors: List[str]) -> str:
    """Create prompt for error code selection."""
    return (
        "You are helping to identify relevant NKI error codes from error output.\n\n"
        f"Here is the error output:\n{error_message}\n\n"
        f"Available error codes:\n{sorted(available_errors)}\n\n"
        "Please identify the most relevant error codes in this output. Return your selection as a JSON list "
        "of error codes (without the 'ERROR: ' prefix). For example: [\"INVALID_TYPE\", \"OUT_OF_BOUNDS\"]\n\n"
        "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
        "I repeat your entire response must be a valid JSON array. Do not deviate from this format"
    )


def create_additional_functions_prompt(current_functions: List[str], error_message: str, 
                                     available_functions: List[str]) -> str:
    """Create prompt for additional function selection."""
    return (
        "Based on the error message below, do we need to include documentation for any additional NKI functions "
        "that weren't selected earlier?\n\n"
        f"Current functions: {', '.join(current_functions)}\n\n"
        f"Error message:\n{error_message}\n\n"
        f"Available functions: {', '.join(available_functions)}\n\n"
        "Return ONLY a JSON list of additional function names needed (without the 'nki_language_' prefix). "
        "If no additional functions are needed, return an empty list [].\n\n"
        "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
    )


def create_change_report_prompt(old_error_message: str, old_error_line_info: str, 
                               reasoning_text: str, error_message: str, 
                               new_error_line_info: str) -> str:
    """Create prompt for change analysis report."""
    return (
        "You are analyzing the results of changes made to fix errors in a NKI kernel.\n\n"
        f"Previous error message:\n{old_error_message}\n\n"
        f"Previous error line information:\n{old_error_line_info}\n\n"
        f"Applied solution (reasoning):\n{reasoning_text}\n\n"
        f"New error message after applying the solution:\n{error_message}\n\n"
        f"New error line information:\n{new_error_line_info}\n\n"
        "Please provide your analysis in the following JSON format:\n"
        "```json\n"
        "{\n"
        " \"correct\": boolean, // true if the fix resolved the initial problem, false otherwise\n"
        " \"report\": \"string\" // brief explanation of why the solution worked or didn't work\n"
        "}\n"
        "```\n\n"
        "The 'correct' field should be true if the exact error we had last time has been fixed."
        "it is still deemed correct even if a different error arises, we are just focusing on the "
        "last error we were trying to fix\n"
        "Remember, if the previous error and the new error are different, that means the solution is correct and should be true"
        "Keep your report brief and focused on the specific changes and their effects. This is important"
        "remember to keep the report consise and focused on key words on why it worked or failed"
    ) 