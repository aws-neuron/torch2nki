import glob
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from extraction import read_file
import json

######################
# Direct Documentation Access Functions
######################

def get_available_functions(docs_dir):
    """
    Get a list of all available NKI functions from the documentation directory.
    
    Args:
        docs_dir (str): Path to the documentation directory
        
    Returns:
        list: A list of function names
    """
    # Get all .txt files in the directory
    file_pattern = os.path.join(docs_dir, "*.txt")
    files = glob.glob(file_pattern)
    
    # Extract function names from filenames
    functions = []
    for file_path in files:
        # Get the basename and remove the prefix and suffix
        basename = os.path.basename(file_path)
        if basename.startswith("nki_language_") and basename.endswith(".txt"):
            # Extract the function name between nki_language_ and .txt
            function_name = basename[len("nki_language_"):-len(".txt")]
            functions.append(function_name)
    
    return sorted(functions)

def select_relevant_functions(llm, user_prompt, available_functions):
    """
    Use LLM to select relevant functions for the task.
    
    Args:
        llm: The LLM instance
        user_prompt (str): The user prompt
        available_functions (list): List of available functions
        
    Returns:
        list: List of selected function names
    """
    function_selection_prompt = ChatPromptTemplate.from_template(
        "You are helping to select relevant NKI functions for a kernel implementation task.\n\n"
        "Here is the task:\n{user_prompt}\n\n"
        "Available functions:\n{function_list}\n\n"
        "Please select the most relevant functions for this task. Return your selection as a JSON list "
        "of function names (without the 'nki_language_' prefix). Choose only what's necessary for the task. "
        "For example: [\"add\", \"multiply\", \"subtract\"]"
    )
    
    # Format function list for display
    function_list = "\n".join(sorted(available_functions))
    
    function_selection_chain = (
        function_selection_prompt 
        | llm 
        | StrOutputParser()
    )
    
    response = function_selection_chain.invoke({
        "user_prompt": user_prompt,
        "function_list": function_list
    })
    
    # Parse the JSON response
    try:
        selected_functions = json.loads(response)
        # Validate that all selected functions are in available_functions
        valid_selections = [f for f in selected_functions if f in available_functions]
        return valid_selections
    except Exception as e:
        print(f"Error parsing selected functions: {e}")
        # Extract function names using regex as fallback
        pattern = re.compile(r'["\']([\w_]+)["\']')
        matches = pattern.findall(response)
        valid_selections = [f for f in matches if f in available_functions]
        return valid_selections

def load_function_documentation(docs_dir, function_names):
    """
    Load documentation for the selected functions.
    
    Args:
        docs_dir (str): Path to the documentation directory
        function_names (list): List of function names to load
        
    Returns:
        str: Combined documentation text
    """
    documentation = []
    
    for function_name in function_names:
        file_path = os.path.join(docs_dir, f"nki_language_{function_name}.txt")
        if os.path.exists(file_path):
            try:
                content = read_file(file_path)
                documentation.append(f"FUNCTION: {function_name}")
                documentation.append("-" * 50)
                documentation.append(content)
                documentation.append("\n" + "=" * 80 + "\n")
            except Exception as e:
                print(f"Error loading documentation for {function_name}: {e}")
    # print(documentation[:50])
    return "\n".join(documentation)
