#TODO seperate into multiple utility files

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pinecone import Pinecone
import os
import re
import subprocess
import traceback
import datetime

# Set up LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "torch2nki"

######################
# NKI Error Parser
######################

class NKIErrorParser:
    """Parser for NKI error messages documentation."""
    
    def __init__(self, error_doc_path):
        """
        Initialize the error parser with the path to the error documentation file.
        
        Args:
            error_doc_path (str): Path to the NKI error messages documentation file
        """
        self.error_doc_path = error_doc_path
        self.error_database = self._parse_error_file()
        
    def _parse_error_file(self):
        """
        Parse the error documentation file and build an error database.
        
        Returns:
            dict: A dictionary mapping error codes to their documentation
        """
        if not os.path.exists(self.error_doc_path):
            print(f"Error documentation file not found: {self.error_doc_path}")
            return {}
            
        try:
            with open(self.error_doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Split the content by the error separator pattern
            error_pattern = re.compile(r'ERROR: ([a-zA-Z0-9_-]+)\s*\n==+\n(.*?)(?=\n==+\nERROR:|$)', re.DOTALL)
            errors = error_pattern.findall(content)
            
            error_database = {}
            for error_code, error_content in errors:
                # Parse instructions and code examples
                instructions = []
                code_examples = []
                
                # Extract instructions
                instruction_pattern = re.compile(r'Instruction (\d+): (.*?)(?=\nInstruction \d+:|Code Example \d+:|$)', re.DOTALL)
                for _, instruction_text in instruction_pattern.findall(error_content):
                    instructions.append(instruction_text.strip())
                
                # Extract code examples
                code_pattern = re.compile(r'Code Example (\d+):(.*?)(?=\nCode Example \d+:|$)', re.DOTALL)
                for _, code_text in code_pattern.findall(error_content):
                    code_examples.append(code_text.strip())
                
                error_database[error_code] = {
                    'instructions': instructions,
                    'code_examples': code_examples,
                    'raw_content': error_content.strip()
                }
                
            return error_database
            
        except Exception as e:
            print(f"Error parsing documentation file: {e}")
            return {}
    
    def get_error_info(self, error_code):
        """
        Get information about a specific error code.
        
        Args:
            error_code (str): The error code to look up
            
        Returns:
            dict: Error information including instructions and code examples
        """
        # Normalize error code by removing "ERROR: " prefix if present
        if error_code.upper().startswith("ERROR: "):
            error_code = error_code[7:]
            
        # Check if we have this error in our database
        if error_code in self.error_database:
            return self.error_database[error_code]
        
        # Try case-insensitive match if exact match fails
        for key in self.error_database.keys():
            if key.lower() == error_code.lower():
                return self.error_database[key]
                
        return None
    
    def list_all_errors(self):
        """
        List all error codes in the database.
        
        Returns:
            list: A list of all error codes
        """
        return list(self.error_database.keys())
    
    def search_errors(self, keyword):
        """
        Search for errors containing a keyword.
        
        Args:
            keyword (str): Keyword to search for in error codes and content
            
        Returns:
            list: A list of matching error codes
        """
        matches = []
        keyword = keyword.lower()
        
        for code, info in self.error_database.items():
            if (keyword in code.lower() or 
                keyword in info['raw_content'].lower()):
                matches.append(code)
                
        return matches

def parse_error_output(error_output, error_parser):
    """
    Parse error output from test script and get detailed error information.
    
    Args:
        error_output (str): The error output from the test script
        error_parser (NKIErrorParser): Instance of the NKI error parser
        
    Returns:
        list: A list of dictionaries containing error codes and their documentation
    """
    # Extract error codes from the output
    error_pattern = re.compile(r'ERROR: ([a-zA-Z0-9_-]+)', re.IGNORECASE)
    error_matches = error_pattern.findall(error_output)
    
    # Get documentation for each error
    error_docs = []
    for error_code in error_matches:
        error_info = error_parser.get_error_info(error_code)
        if error_info:
            error_docs.append({
                'code': error_code,
                'info': error_info
            })
    
    return error_docs

def format_error_docs(error_docs):
    """
    Format error documentation for display.
    
    Args:
        error_docs (list): List of error documentations
        
    Returns:
        str: Formatted error documentation
    """
    if not error_docs:
        return "No documented errors found in the output."
        
    output = []
    
    for doc in error_docs:
        output.append(f"ERROR: {doc['code']}")
        output.append("=" * 50)
        
        error_info = doc['info']
        
        # Add instructions
        if error_info['instructions']:
            for i, instruction in enumerate(error_info['instructions'], 1):
                output.append(f"Instruction {i}: {instruction}")
            
        # Add code examples
        if error_info['code_examples']:
            for i, example in enumerate(error_info['code_examples'], 1):
                output.append(f"Code Example {i}:")
                output.append(example)
                
        output.append("")
        output.append("=" * 50)
        
    return "\n".join(output)

######################
# Extraction and Utility Functions
######################

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

######################
# LangChain RAG Functions
######################

def setup_rag_components(pinecone_api_key, pinecone_index_name):
    """Set up and return the RAG components."""
    # Initialize LLMs
    query_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    kernel_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    # Set up vector store and retriever with improved error handling
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index instance
        index = pc.Index(name=pinecone_index_name)
        
        # Check for namespaces in the index
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        active_namespace = namespaces[0] if namespaces else None
        
        print(f"Index contains {stats.get('total_vector_count', 0)} vectors")
        
        if active_namespace:
            print(f"Using namespace: {active_namespace}")
            # Create the vector store using the index with namespace
            vectorstore = PineconeVectorStore(
                embedding=embeddings,
                index=index,
                namespace=active_namespace
            )
        else:
            # Create the vector store without namespace
            vectorstore = PineconeVectorStore(
                embedding=embeddings,
                index=index
            )
        
        # Create retriever with increased k to ensure we get results
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Increased from 2 to 5 to get more results
            
        )
        
        # Test the retriever with a simple query to validate it works
        test_results = retriever.invoke("language apis")
        if test_results:
            print(f"Successfully connected to Pinecone and retrieved {len(test_results)} documents")
        else:
            print("Connected to Pinecone but retrieval returned no results - continuing anyway")
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        print("Falling back to dummy retriever")
        # Create a dummy retriever that returns empty results
        class DummyRetriever:
            def invoke(self, _):
                return []
        
        retriever = DummyRetriever()
    
    # Create the query generation chain
    query_generation_prompt = ChatPromptTemplate.from_template(
        "Identify key technical concepts for NKI kernel. Be brief (max 100 words).\n\n"
        "What technical concepts should I retrieve for this kernel task? Use a bullet point list of different functions / methods that could be helpful"
        "Specifically, make a list of specific vector operations you want to implement.{user_prompt}"
    )
    
    query_generation_chain = (
        query_generation_prompt 
        | query_llm 
        | StrOutputParser()
    )
    
    return query_generation_chain, retriever, kernel_llm

def format_context(docs):
    """Format the retrieved documents into a context string."""
    context = ""
    for i, doc in enumerate(docs):
        context += f"Doc{i+1}: "
        
        # Get content
        content = doc.page_content
        metadata = doc.metadata
        
        # Get title from metadata if available
        title = metadata.get('title', 'No title')
        
        # Check if content is too long
        if len(content) > 500:
            content = content[:500] + "..."
            
        context += f"{title} - {content}\n\n"
        
    if not context:
        context = "No relevant documents found."
        
    return context

def perform_rag_retrieval(query_generation_chain, retriever, user_prompt, error_message=None, trace_log_path=None):
    """
    Perform RAG retrieval based on user prompt and error message.
    
    Args:
        query_generation_chain: The query generation chain
        retriever: The document retriever
        user_prompt: The original user prompt
        error_message: Optional error message to include in the query generation
        trace_log_path: Optional path to log the retrieval process
        
    Returns:
        str: The formatted context from retrieved documents
    """
    # Construct a combined prompt including error message if available
    combined_prompt = user_prompt
    if error_message:
        # Extract only the relevant parts of the error message (first 200 chars)
        error_summary = error_message[:200] + "..." if len(error_message) > 200 else error_message
        combined_prompt += f"\n\nAdditional context - Error encountered: {error_summary}"
    
    # Log what we're doing
    if trace_log_path:
        log_to_file(trace_log_path, "GENERATING RETRIEVAL QUERY...")
        log_to_file(trace_log_path, f"COMBINED PROMPT FOR QUERY:\n{combined_prompt}\n")
    
    print("Generating retrieval query...")
    retrieval_query = query_generation_chain.invoke({"user_prompt": combined_prompt})
    
    print(f"Query generated: {retrieval_query}...")
    if trace_log_path:
        log_to_file(trace_log_path, f"GENERATED QUERY:\n{retrieval_query}\n")
    
    print("Retrieving relevant documents...")
    if trace_log_path:
        log_to_file(trace_log_path, "RETRIEVING DOCUMENTS FROM PINECONE...")
    
    docs = retriever.invoke(retrieval_query)
    context = format_context(docs)
    
    if trace_log_path:
        log_to_file(trace_log_path, f"RETRIEVED CONTEXT:\n{context}\n")
    
    return context, retrieval_query

def generate_kernel_with_rag_and_error_loop(
    system_prompt_path, 
    user_prompt_path, 
    output_address,
    kernel_module_path,
    test_script_path,
    test_script_output,
    reasoning_log_path,
    error_doc_path,  # New parameter for error documentation file
    pinecone_api_key, 
    pinecone_index_name,
    max_iterations=5
):
    """
    Generate a NKI kernel using RAG with LangChain and iteratively improve it
    based on error feedback with detailed error documentation.
    """
    print("Initializing LangChain components...")
    
    # Initialize the error parser
    print(f"Initializing NKI error parser from {error_doc_path}")
    error_parser = NKIErrorParser(error_doc_path)
    print(f"Loaded {len(error_parser.list_all_errors())} error codes from documentation")
    
    # Set up detailed trace log file
    trace_log_path = output_address + ".detailed_trace.txt"
    log_to_file(trace_log_path, "=== DETAILED TRACE LOG ===", append=False)
    log_to_file(trace_log_path, f"Starting new kernel generation process at {datetime.datetime.now()}")
    
    # Load the initial prompts
    system_prompt = read_file(system_prompt_path)
    user_prompt = read_file(user_prompt_path)
    
    log_to_file(trace_log_path, f"System Prompt:\n{system_prompt}\n")
    log_to_file(trace_log_path, f"User Prompt:\n{user_prompt}\n")
    
    print(f"Starting RAG process for: {user_prompt[:50]}...")
    
    # Set up RAG components
    query_generation_chain, retriever, kernel_llm = setup_rag_components(
        pinecone_api_key, pinecone_index_name
    )
    
    # Initial RAG-based kernel generation
    try:
        # Perform initial RAG retrieval
        context, retrieval_query = perform_rag_retrieval(
            query_generation_chain,
            retriever,
            user_prompt,
            trace_log_path=trace_log_path
        )
        
        # Log the query and retrieval process
        with open(output_address + ".query_log", "w") as f:
            f.write(f"USER PROMPT:\n{user_prompt}\n\n")
            f.write(f"GENERATED QUERY:\n{retrieval_query}\n\n")
            f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
        
        print(f"Query and context saved to {output_address}.query_log")
        
        # Initial kernel generation with RAG
        print("Generating initial kernel...")
        log_to_file(trace_log_path, "GENERATING INITIAL KERNEL...")
        
        initial_generation_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Task: {user_prompt}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Generate a NKI kernel for the task."
        )
        
        # Log the full prompt being sent to the LLM
        full_prompt = initial_generation_prompt.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context
        )
        log_to_file(trace_log_path, f"FULL PROMPT TO LLM:\n{full_prompt}\n")
        
        initial_kernel_chain = (
            initial_generation_prompt 
            | kernel_llm 
            | StrOutputParser()
        )
        
        initial_generation = initial_kernel_chain.invoke({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context
        })
        
        # Save raw output
        write_file(output_address, initial_generation)
        print(f"Raw LLM output saved to {output_address}")
        log_to_file(trace_log_path, f"LLM RESPONSE:\n{initial_generation}\n")
        
        # Extract the kernel code
        try:
            kernel_code = extract_kernel_from_llm_response(initial_generation)
            write_file(kernel_module_path, kernel_code)
            print(f"Initial kernel code saved to {kernel_module_path}")
            log_to_file(trace_log_path, f"EXTRACTED KERNEL CODE:\n{kernel_code}\n")
        except ValueError as e:
            error_msg = f"Error extracting kernel code: {e}"
            print(error_msg)
            log_to_file(trace_log_path, error_msg)
            return
        
        # Create enhanced error re-injection prompt with error documentation
        enhanced_error_reinject_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Here is the kernel you just wrote:\n"
            "--------------------------------------------------\n"
            "{kernel_code}\n"
            "--------------------------------------------------\n\n"
            "Here is the error message it got:\n"
            "--------------------------------------------------\n"
            "{error_message}\n"
            "--------------------------------------------------\n\n"
            "Here is detailed documentation about the specific errors encountered:\n"
            "--------------------------------------------------\n"
            "{error_documentation}\n"
            "--------------------------------------------------\n\n"
            "Try to fix it. Clearly explain your line of reasoning as well as what you think the error is, and how you plan to fix it. "
            "Clearly put your initial reasoning inside triple stars like this *** example: i am making this change because i love unicorns ***. "
            "I want all your initial reasoning inside of these triple stars, not just the summary at the end.\n\n"
            "Retrieved Context:\n{context}\n\n"
        )
        
        enhanced_error_chain = (
            enhanced_error_reinject_prompt 
            | kernel_llm 
            | StrOutputParser()
        )
        
        # Iterative error correction loop
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            log_to_file(trace_log_path, f"\n=== ITERATION {iteration + 1} ===\n")
            
            # Run the test script and get error output
            log_to_file(trace_log_path, f"RUNNING TEST SCRIPT: {test_script_path}")
            error_message = run_script_and_save_output(test_script_path, test_script_output)
            log_to_file(trace_log_path, f"TEST SCRIPT OUTPUT:\n{error_message}\n")
            
            # If no errors, we're done
            if "Error" not in error_message and "error" not in error_message and "ERROR" not in error_message:
                print("No errors detected! Kernel generation successful.")
                log_to_file(trace_log_path, "NO ERRORS DETECTED. KERNEL GENERATION SUCCESSFUL.")
                break
            
            # Parse error message and get documentation
            print("Parsing error message for detailed documentation...")
            error_docs = parse_error_output(error_message, error_parser)
            error_documentation = format_error_docs(error_docs)
            
            # Log the parsed error documentation
            log_to_file(trace_log_path, f"PARSED ERROR DOCUMENTATION:\n{error_documentation}\n")
            print(f"Found {len(error_docs)} documented errors in the output")
            
            # Save error documentation to a separate file for this iteration
            error_doc_file = f"{output_address}.error_doc.iteration{iteration+1}.txt"
            with open(error_doc_file, "w", encoding="utf-8") as f:
                f.write(error_documentation)
            print(f"Error documentation saved to {error_doc_file}")
            
            # If no documented errors found, use a fallback message
            if not error_docs:
                error_documentation = "No specific documentation found for the errors in the output. Please analyze the error message carefully."
            
            # Perform RAG retrieval again with the error message to get updated context
            print(f"Performing RAG retrieval for iteration {iteration + 1}...")
            log_to_file(trace_log_path, f"PERFORMING RAG RETRIEVAL FOR ITERATION {iteration + 1}...")
            
            updated_context, updated_query = perform_rag_retrieval(
                query_generation_chain,
                retriever,
                user_prompt,
                error_message=error_message,  # Include the error message in the query generation
                trace_log_path=trace_log_path
            )
            
            # Log the updated query and retrieval
            iteration_query_log = f"{output_address}.query_log.iteration{iteration+1}"
            with open(iteration_query_log, "w") as f:
                f.write(f"ITERATION {iteration + 1} QUERY:\n\n")
                f.write(f"ERROR MESSAGE:\n{error_message}\n\n")
                f.write(f"GENERATED QUERY:\n{updated_query}\n\n")
                f.write(f"RETRIEVED CONTEXT:\n{updated_context}\n\n")
            
            print(f"Updated query and context saved to {iteration_query_log}")
            
            # Generate improved kernel with error feedback, documentation, and updated context
            print(f"Generating improved kernel (iteration {iteration + 1})...")
            log_to_file(trace_log_path, f"GENERATING IMPROVED KERNEL (ITERATION {iteration + 1})...")
            
            # Log the full error prompt being sent to the LLM
            full_error_prompt = enhanced_error_reinject_prompt.format(
                system_prompt=system_prompt,
                kernel_code=read_file(kernel_module_path),
                error_message=error_message,
                error_documentation=error_documentation,
                context=updated_context  # Use the updated context
            )
            log_to_file(trace_log_path, f"FULL ERROR PROMPT TO LLM:\n{full_error_prompt}\n")
            
            improved_generation = enhanced_error_chain.invoke({
                "system_prompt": system_prompt,
                "kernel_code": read_file(kernel_module_path),
                "error_message": error_message,
                "error_documentation": error_documentation,
                "context": updated_context  # Use the updated context
            })
            
            # Save the raw output
            write_file(output_address, improved_generation)
            print(f"Raw LLM output saved to {output_address}")
            log_to_file(trace_log_path, f"LLM RESPONSE FOR ITERATION {iteration + 1}:\n{improved_generation}\n")
            
            # Extract reasoning and log it
            reasoning_text = extract_reasoning(improved_generation)
            if reasoning_text:
                with open(reasoning_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"=== Iteration {iteration + 1} ===\n")
                    log_file.write(reasoning_text)
                    log_file.write("\n\n")
                print("Reasoning extracted and appended to reasoning log.")
                log_to_file(trace_log_path, f"EXTRACTED REASONING:\n{reasoning_text}\n")
            else:
                print("No reasoning found in the output.")
                log_to_file(trace_log_path, "NO REASONING FOUND IN THE OUTPUT.")
            
            # Extract the updated kernel code
            try:
                kernel_code = extract_kernel_from_llm_response(improved_generation)
                write_file(kernel_module_path, kernel_code)
                print(f"Updated kernel code saved to {kernel_module_path}")
                log_to_file(trace_log_path, f"UPDATED KERNEL CODE:\n{kernel_code}\n")
            except ValueError as e:
                error_msg = f"Error extracting kernel code: {e}"
                print(error_msg)
                log_to_file(trace_log_path, error_msg)
                continue
            
            # Pause for review before the next iteration if needed
            if iteration < max_iterations - 1:
                log_to_file(trace_log_path, "WAITING FOR USER INPUT TO CONTINUE TO NEXT ITERATION...")
                input("Press Enter to continue to the next iteration (or Ctrl+C to exit)...")
        
        print("Kernel generation process completed.")
        log_to_file(trace_log_path, "KERNEL GENERATION PROCESS COMPLETED.")
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in kernel generation pipeline: {e}")
        log_to_file(trace_log_path, f"ERROR IN KERNEL GENERATION PIPELINE:\n{e}\n{error_details}")
        
        # Save the error
        with open(output_address, "w") as f:
            f.write(f"Error generating kernel: {str(e)}\n\n{error_details}")
        
        with open(output_address + ".query_log", "a") as f:
            f.write(f"\nPIPELINE ERROR:\n{error_details}")


if __name__ == "__main__":
    # Define constant file paths
    system_prompt_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/prompts/system_prompt_for_rag.txt"
    user_prompt_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/prompts/user_prompt_for_rag.txt"
    output_address = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/generation/samples/vector_add.txt"  # Raw OpenAI output
    kernel_module_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/generation/samples/vector_add_kernel.py"  # Kernel module file
    test_script_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/evaluation/samples/test_vector_add.py"
    test_script_output = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/prompts/script_output.txt"
    reasoning_log_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/generation/samples/reasoning_log.txt"
    
    # Add path to error documentation
    error_doc_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/documentation/nki_documentation/nki_error_messages.txt"
    
    # Get credentials
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')
    
    if not pinecone_api_key or not pinecone_index_name:
        print("Error: Environment variables PINECONE_API_KEY and PINECONE_INDEX_NAME must be set.")
        exit(1)
    
    if not os.environ.get('LANGCHAIN_API_KEY') and os.environ.get('LANGCHAIN_TRACING_V2') == 'true':
        print("Warning: LANGCHAIN_API_KEY not set. Tracing will be disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Run the enhanced generator with error loop
    generate_kernel_with_rag_and_error_loop(
        system_prompt_path,
        user_prompt_path,
        output_address,
        kernel_module_path,
        test_script_path,
        test_script_output,
        reasoning_log_path,
        error_doc_path,  # New parameter
        pinecone_api_key,
        pinecone_index_name,
        max_iterations=5
    )