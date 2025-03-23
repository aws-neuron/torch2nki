from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import sys
import traceback
import datetime


# Now you can import using relative imports

from rag_funcs import setup_rag_components, format_context
from extraction import extract_kernel_from_llm_response, extract_reasoning, run_script_and_save_output, read_file, write_file, log_to_file

def generate_kernel_with_rag_and_error_loop(
    system_prompt_path, 
    user_prompt_path, 
    output_address,
    kernel_module_path,
    test_script_path,
    test_script_output,
    reasoning_log_path,
    pinecone_api_key, 
    pinecone_index_name,
    max_iterations=5
):
    """
    Generate a NKI kernel using RAG with LangChain and iteratively improve it
    based on error feedback.
    """
    print("Initializing LangChain components...")
    
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
        print("Generating retrieval query...")
        log_to_file(trace_log_path, "GENERATING RETRIEVAL QUERY...")
        retrieval_query = query_generation_chain.invoke({"user_prompt": user_prompt})
        
        print(f"Query generated: {retrieval_query[:100]}...")
        log_to_file(trace_log_path, f"GENERATED QUERY:\n{retrieval_query}\n")
        
        print("Retrieving relevant documents...")
        log_to_file(trace_log_path, "RETRIEVING DOCUMENTS FROM PINECONE...")
        docs = retriever.invoke(retrieval_query)
        context = format_context(docs)
        
        log_to_file(trace_log_path, f"RETRIEVED CONTEXT:\n{context}\n")
        
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
        
        # Create error re-injection prompt
        error_reinject_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Here is the kernel you just wrote:\n"
            "--------------------------------------------------\n"
            "{kernel_code}\n"
            "--------------------------------------------------\n\n"
            "Here is the error message it got:\n"
            "--------------------------------------------------\n"
            "{error_message}\n"
            "--------------------------------------------------\n\n"
            "Try to fix it. Clearly explain your line of reasoning as well as what you think the error is, and how you plan to fix it. "
            "Clearly put your initial reasoning inside triple stars like this *** example: i am making this change because i love unicorns ***. "
            "I want all your initial reasoning inside of these triple stars, not just the summary at the end.\n\n"
            "Retrieved Context:\n{context}\n\n"
        )
        
        error_chain = (
            error_reinject_prompt 
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
            
            # Generate improved kernel with error feedback
            print(f"Generating improved kernel (iteration {iteration + 1})...")
            log_to_file(trace_log_path, f"GENERATING IMPROVED KERNEL (ITERATION {iteration + 1})...")
            
            # Log the full error prompt being sent to the LLM
            full_error_prompt = error_reinject_prompt.format(
                system_prompt=system_prompt,
                kernel_code=read_file(kernel_module_path),
                error_message=error_message,
                context=context
            )
            log_to_file(trace_log_path, f"FULL ERROR PROMPT TO LLM:\n{full_error_prompt}\n")
            
            improved_generation = error_chain.invoke({
                "system_prompt": system_prompt,
                "kernel_code": read_file(kernel_module_path),
                "error_message": error_message,
                "context": context
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
    
    # Get credentials
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')
    
    if not pinecone_api_key or not pinecone_index_name:
        print("Error: Environment variables PINECONE_API_KEY and PINECONE_INDEX_NAME must be set.")
        exit(1)
    
    if not os.environ.get('LANGCHAIN_API_KEY') and os.environ.get('LANGCHAIN_TRACING_V2') == 'true':
        print("Warning: LANGCHAIN_API_KEY not set. Tracing will be disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Run the generator with error loop
    generate_kernel_with_rag_and_error_loop(
        system_prompt_path,
        user_prompt_path,
        output_address,
        kernel_module_path,
        test_script_path,
        test_script_output,
        reasoning_log_path,
        pinecone_api_key,
        pinecone_index_name,
        max_iterations=5
    )