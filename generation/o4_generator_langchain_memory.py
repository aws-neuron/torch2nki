from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone
import os
import re
import subprocess
import time
import traceback
import datetime

# Set up LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "torch2nki"

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
    """Set up and return the RAG components with memory."""
    # Initialize LLMs
    query_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    kernel_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7
    )
    
    # Initialize memory for the kernel development process
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
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
        "What technical concepts should I retrieve for this kernel task? {user_prompt}"
    )
    
    query_generation_chain = (
        query_generation_prompt 
        | query_llm 
        | StrOutputParser()
    )
    
    return query_generation_chain, retriever, kernel_llm, memory

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
    based on error feedback, while maintaining memory of previous iterations.
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
    
    # Set up RAG components with memory
    query_generation_chain, retriever, kernel_llm, memory = setup_rag_components(
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
        
        # Create a memory-enabled prompt for initial generation
        initial_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "Task: {user_prompt}\n\nRetrieved Context:\n{context}\n\nGenerate a NKI kernel for the task.")
        ])
        
        # Log the full prompt being sent to the LLM
        full_prompt = initial_generation_prompt.format_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context
        )
        log_to_file(trace_log_path, f"FULL PROMPT TO LLM:\n{full_prompt}\n")
        
        # Create the initial chain with memory
        initial_kernel_chain = initial_generation_prompt | kernel_llm | StrOutputParser()
        
        initial_generation = initial_kernel_chain.invoke({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context
        })
        
        # Add the initial interaction to memory
        memory.chat_memory.add_user_message(f"Task: {user_prompt}\n\nRetrieved Context:\n{context}\n\nGenerate a NKI kernel for the task.")
        memory.chat_memory.add_ai_message(initial_generation)
        
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
        
        # Create error re-injection prompt with memory
        error_reinject_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", """Here is the kernel you just wrote:
--------------------------------------------------
{kernel_code}
--------------------------------------------------

Here is the error message it got:
--------------------------------------------------
{error_message}
--------------------------------------------------

Try to fix it. Clearly explain your line of reasoning as well as what you think the error is, and how you plan to fix it.
Clearly put your initial reasoning inside triple stars like this *** example: i am making this change because i love unicorns ***.
I want all your initial reasoning inside of these triple stars, not just the summary at the end.

Retrieved Context:
{context}

{chat_history}""")
        ])
        
        # Create the error chain with memory
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
            
            # Get chat history from memory
            chat_history = memory.load_memory_variables({})["chat_history"]
            formatted_chat_history = "Previous iterations:\n"
            for i, message in enumerate(chat_history):
                if i > 0:  # Skip the first system message if present
                    prefix = "Human: " if isinstance(message, HumanMessage) else "AI: "
                    # Truncate long messages
                    content = message.content
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    formatted_chat_history += f"{prefix}{content}\n\n"
            
            # Log the full error prompt being sent to the LLM
            full_error_prompt = error_reinject_prompt.format_messages(
                system_prompt=system_prompt,
                kernel_code=read_file(kernel_module_path),
                error_message=error_message,
                context=context,
                chat_history=formatted_chat_history
            )
            log_to_file(trace_log_path, f"FULL ERROR PROMPT TO LLM:\n{full_error_prompt}\n")
            
            improved_generation = error_chain.invoke({
                "system_prompt": system_prompt,
                "kernel_code": read_file(kernel_module_path),
                "error_message": error_message,
                "context": context,
                "chat_history": formatted_chat_history
            })
            
            # Add to memory
            user_error_message = f"""Here is the kernel you just wrote:
--------------------------------------------------
{read_file(kernel_module_path)}
--------------------------------------------------

Here is the error message it got:
--------------------------------------------------
{error_message}
--------------------------------------------------

Try to fix it."""
            
            memory.chat_memory.add_user_message(user_error_message)
            memory.chat_memory.add_ai_message(improved_generation)
            
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