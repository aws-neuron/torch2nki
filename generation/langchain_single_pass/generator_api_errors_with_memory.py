from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import os
import re
import traceback
import datetime
import json

from extraction import extract_kernel_from_llm_response, extract_reasoning, run_script_and_save_output, read_file, write_file, log_to_file
from doc_grabber import get_available_functions, select_relevant_functions, load_function_documentation
from nki_error_parsing import NKIErrorParser, extract_error_details, get_available_error_codes, select_relevant_errors, load_error_documentation


def generate_kernel_with_direct_docs_and_error_loop(
    system_prompt_path, 
    user_prompt_path, 
    output_address,
    kernel_module_path,
    test_script_path,
    test_script_output,
    reasoning_log_path,
    error_doc_path,
    docs_dir,
    max_iterations=15
):
    """
    Generate a NKI kernel using direct function documentation access and iteratively 
    improve it based on error feedback with detailed error documentation.
    Now with LangChain memory to maintain context between iterations.
    """
    print("Initializing components...")
    
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
    
    print(f"Starting documentation-based generation for: {user_prompt[:50]}...")
    
    # Initialize LLMs
    query_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    kernel_llm = ChatBedrock(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        model_kwargs={"temperature": 0.85},
        region_name="us-west-2"
    )

    # Initialize memory for the main kernel generation conversation
    kernel_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Get list of available functions
    available_functions = get_available_functions(docs_dir)
    print(f"Found {len(available_functions)} available NKI functions in documentation")
    log_to_file(trace_log_path, f"AVAILABLE FUNCTIONS:\n{', '.join(available_functions)}\n")
    
    # Initial kernel generation with direct documentation
    try:
        # Select relevant functions
        print("Selecting relevant functions for the task...")
        log_to_file(trace_log_path, "SELECTING RELEVANT FUNCTIONS...")
        
        selected_functions = select_relevant_functions(
            query_llm,
            user_prompt,
            available_functions
        )
        
        print(f"Selected functions: {', '.join(selected_functions)}")
        log_to_file(trace_log_path, f"SELECTED FUNCTIONS:\n{', '.join(selected_functions)}\n")
        
        # Load documentation for selected functions
        print("Loading documentation for selected functions...")
        log_to_file(trace_log_path, "LOADING FUNCTION DOCUMENTATION...")
        
        function_docs = load_function_documentation(docs_dir, selected_functions)
        log_to_file(trace_log_path, f"LOADED DOCUMENTATION:\n{function_docs[:500]}...\n")
        
        # Log the selected functions and their documentation
        with open(output_address + ".function_selection", "w") as f:
            f.write(f"USER PROMPT:\n{user_prompt}\n\n")
            f.write(f"SELECTED FUNCTIONS:\n{', '.join(selected_functions)}\n\n")
            f.write(f"FUNCTION DOCUMENTATION:\n{function_docs}\n\n")
        
        print(f"Function selection and documentation saved to {output_address}.function_selection")
        
        # Initial kernel generation with function documentation
        print("Generating initial kernel...")
        log_to_file(trace_log_path, "GENERATING INITIAL KERNEL...")
        
        # First message to memory is the system prompt
        kernel_memory.chat_memory.add_message(SystemMessage(content=system_prompt))
        
        # Add the task and documentation as a user message
        initial_prompt = f"Task: {user_prompt}\n\nFunction Documentation:\n{function_docs}\n\nGenerate a NKI kernel for the task."
        kernel_memory.chat_memory.add_message(HumanMessage(content=initial_prompt))
        
        # Log the full prompt being sent to the LLM
        log_to_file(trace_log_path, f"FULL PROMPT TO LLM:\n{system_prompt}\n\n{initial_prompt}\n")
        
        # Generate the initial response
        initial_generation = kernel_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_prompt)
        ]).content
        
        # Add the LLM's response to memory
        kernel_memory.chat_memory.add_message(AIMessage(content=initial_generation))
        
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
        
        # Set up the error reinject prompt template with memory
        enhanced_error_reinject_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content=(
                "Previous error message:\n"
                "--------------------------------------------------\n"
                "{previous_error_message}\n"
                "--------------------------------------------------\n\n"
                "Function Documentation:\n"
                "--------------------------------------------------\n"
                "{function_docs}\n"
                "--------------------------------------------------\n\n"
                "Generate a new improved kernel for this task. Clearly explain your line of reasoning in one sentence, trying "
                "to keep it as brief as possible. Focus on explaining what solution you are planning on using to "
                "fix the error. Remember to keep it concise, but explanatory as you will be referencing this later to make sure "
                "you are not trying to do the same fixes multiple times. "
                "Your output should include the entire kernel code, NOT just individual fixes. I want to be able to run the code inside the ``` ```"
                "The way I want your response structured is an explanation of your reasoning at the very start inside *** *** triple stars. "
                "Then, immediatly after write the python nki code inside triple backticks ``` ```."
                "I repeat, I only want your output to first be the line of reasoning inside triple stars, then the "
                "nki kernel code inside triple backticks. Do NOT put the reasoning inside the nki kernel code."
            ))
        ])
        
        # Variable to store the previous error message
        previous_error_message = ""
        
        # Iterative error correction loop
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            log_to_file(trace_log_path, f"\n=== ITERATION {iteration + 1} ===\n")
            
            # Store the previous error message before running any new tests
            old_error_message = previous_error_message if 'previous_error_message' in locals() else ""
            
            # Run the test script only if this is iteration 0 (initial code) or after we've generated new code
            # For the first iteration, we need to run the script on the initial code
            if iteration == 0:
                # Run the test script and get error output for the initial kernel
                log_to_file(trace_log_path, f"RUNNING TEST SCRIPT ON INITIAL CODE: {test_script_path}")
                error_message = run_script_and_save_output(test_script_path, test_script_output)
                log_to_file(trace_log_path, f"TEST SCRIPT OUTPUT:\n{error_message}\n")
                previous_error_message = error_message
                
                # If no errors in the initial code, we're done
                if "Error" not in error_message and "error" not in error_message and "ERROR" not in error_message:
                    print("No errors detected in initial kernel! Kernel generation successful.")
                    log_to_file(trace_log_path, "NO ERRORS DETECTED IN INITIAL KERNEL. KERNEL GENERATION SUCCESSFUL.")
                    break

            error_line, error_description = extract_error_details(error_message)
            if error_line and error_description:
                print(f"\nERROR LINE: {error_line}")
                print(f"ERROR DESCRIPTION: {error_description}")
                log_to_file(trace_log_path, f"ERROR LINE: {error_line}\n")
                log_to_file(trace_log_path, f"ERROR DESCRIPTION: {error_description}\n")
            else:
                print("\nCould not extract specific error details.")
                log_to_file(trace_log_path, "COULD NOT EXTRACT SPECIFIC ERROR DETAILS.\n")

            # If we've reached here, there were errors in the previous iteration
            # Parse error message and get documentation using API-style approach
            print("Parsing error message for detailed documentation...")
            log_to_file(trace_log_path, "PARSING ERROR MESSAGE...")

            # Get all available error codes
            available_errors = get_available_error_codes(error_parser)
            log_to_file(trace_log_path, f"AVAILABLE ERRORS:\n{', '.join(available_errors)}\n")

            # Select relevant errors using the LLM
            error_selection_prompt = ChatPromptTemplate.from_template(
                "You are helping to identify relevant NKI error codes from error output.\n\n"
                "Here is the error output:\n{error_message}\n\n"
                "Available error codes:\n{error_list}\n\n"
                "Please identify the most relevant error codes in this output. Return your selection as a JSON list "
                "of error codes (without the 'ERROR: ' prefix). For example: [\"INVALID_TYPE\", \"OUT_OF_BOUNDS\"]\n\n"
                "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
                "I repeat your entire response must be a valid JSON array. Do not deviate from this format"
            )

            # Format error list for display
            error_list = "\n".join(sorted(available_errors))

            error_selection_chain = (
                error_selection_prompt
                | query_llm
                | StrOutputParser()
            )

            error_response = error_selection_chain.invoke({
                "error_message": previous_error_message,
                "error_list": error_list
            })

            # Helper function to extract JSON array from text
            def extract_json_array(text):
                # Remove any non-JSON text before or after the array
                text = text.strip()
                # If text begins with characters before [, remove them
                if '[' in text and text[0] != '[':
                    text = text[text.find('['):]
                # If text has characters after the closing ], remove them
                if ']' in text and text[-1] != ']':
                    text = text[:text.rfind(']')+1]
                # If we still don't have a valid JSON looking text, try regex
                if not (text.startswith('[') and text.endswith(']')):
                    import re
                    json_pattern = re.compile(r'\[.*?\]', re.DOTALL)
                    json_match = json_pattern.search(text)
                    if json_match:
                        text = json_match.group(0)
                return text

            # Clean up and parse the response
            try:
                # Clean the response and try to parse it
                cleaned_response = extract_json_array(error_response)
                
                # Handle empty lists represented as empty string, "[]", etc.
                if not cleaned_response or cleaned_response.isspace():
                    selected_errors = []
                elif cleaned_response == "[]":
                    selected_errors = []
                else:
                    selected_errors = json.loads(cleaned_response)
                
                # Validate that all selected errors are in available_errors
                selected_errors = [e for e in selected_errors if e in available_errors]
                
            except Exception as e:
                print(f"Error parsing selected errors: {e}")
                log_to_file(trace_log_path, f"ERROR PARSING SELECTED ERRORS: {e}\n")
                
                # Fallback mechanism: try to extract error codes using regex
                try:
                    pattern = re.compile(r'["\']([\w_-]+)["\']')
                    matches = pattern.findall(error_response)
                    selected_errors = [e for e in matches if e in available_errors]
                    print(f"Using fallback: Extracted errors via regex: {', '.join(selected_errors)}")
                    log_to_file(trace_log_path, f"FALLBACK: EXTRACTED ERRORS VIA REGEX: {', '.join(selected_errors)}\n")
                except Exception as fallback_error:
                    print(f"Fallback parsing also failed: {fallback_error}")
                    log_to_file(trace_log_path, f"FALLBACK PARSING ALSO FAILED: {fallback_error}\n")
                    selected_errors = []

            print(f"Selected errors: {', '.join(selected_errors)}")
            log_to_file(trace_log_path, f"SELECTED ERRORS:\n{', '.join(selected_errors)}\n")

            # Load documentation for selected errors
            error_documentation = load_error_documentation(error_parser, selected_errors)
            log_to_file(trace_log_path, f"LOADED ERROR DOCUMENTATION:\n{error_documentation[:500]}...\n")

            # Log the selected errors and their documentation
            with open(f"{output_address}.error_selection", "w") as f:
                f.write(f"ERROR MESSAGE:\n{previous_error_message}\n\n")
                f.write(f"SELECTED ERRORS:\n{', '.join(selected_errors)}\n\n")
                f.write(f"ERROR DOCUMENTATION:\n{error_documentation}\n\n")

            print(f"Error selection and documentation saved to {output_address}.error_selection")

            # If no documented errors found, use a fallback message
            if not selected_errors:
                error_documentation = "No specific documentation found for the errors in the output. Please analyze the error message carefully."

            # Check if we need additional functions based on error
            print("Checking if additional functions are needed based on error...")
            
            additional_functions_prompt = ChatPromptTemplate.from_template(
                "Based on the error message below, do we need to include documentation for any additional NKI functions "
                "that weren't selected earlier?\n\n"
                "Current functions: {current_functions}\n\n"
                "Error message:\n{error_message}\n\n"
                "Available functions: {all_functions}\n\n"
                "Return ONLY a JSON list of additional function names needed (without the 'nki_language_' prefix). "
                "If no additional functions are needed, return an empty list [].\n\n"
                "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
            )

            additional_functions_chain = (
                additional_functions_prompt 
                | query_llm 
                | StrOutputParser()
            )

            additional_response = additional_functions_chain.invoke({
                "current_functions": ", ".join(selected_functions),
                "error_message": previous_error_message,
                "all_functions": ", ".join(available_functions)
            })

            try:
                # Clean the response and try to parse it
                cleaned_response = extract_json_array(additional_response)
                
                # Handle empty lists represented as empty string, "[]", etc.
                if not cleaned_response or cleaned_response.isspace():
                    additional_functions = []
                elif cleaned_response == "[]":
                    additional_functions = []
                else:
                    additional_functions = json.loads(cleaned_response)
                
                # Only include valid functions that weren't already selected
                new_functions = [f for f in additional_functions 
                            if f in available_functions and f not in selected_functions]
                
                if new_functions:
                    print(f"Adding additional functions: {', '.join(new_functions)}")
                    log_to_file(trace_log_path, f"ADDING ADDITIONAL FUNCTIONS: {', '.join(new_functions)}\n")
                    
                    # Add to selected functions
                    selected_functions.extend(new_functions)
                    
                    # Update function documentation
                    additional_docs = load_function_documentation(docs_dir, new_functions)
                    function_docs += "\n\n" + additional_docs
                    
                    # Log updated documentation
                    with open(f"{output_address}.function_selection", "w") as f:
                        f.write(f"UPDATED SELECTED FUNCTIONS:\n{', '.join(selected_functions)}\n\n")
                        f.write(f"ADDED FUNCTIONS:\n{', '.join(new_functions)}\n\n")
                        f.write(f"ADDED DOCUMENTATION:\n{additional_docs}\n\n")
            except Exception as e:
                print(f"Error parsing additional functions: {e}")
                log_to_file(trace_log_path, f"ERROR PARSING ADDITIONAL FUNCTIONS: {e}\n")
                
                # Fallback mechanism: try to extract function names using regex
                try:
                    pattern = re.compile(r'["\']([\w_]+)["\']')
                    matches = pattern.findall(additional_response)
                    valid_matches = [f for f in matches if f in available_functions and f not in selected_functions]
                    
                    if valid_matches:
                        print(f"Using fallback: Adding functions detected via regex: {', '.join(valid_matches)}")
                        log_to_file(trace_log_path, f"FALLBACK: ADDING FUNCTIONS VIA REGEX: {', '.join(valid_matches)}\n")
                        
                        # Add to selected functions
                        selected_functions.extend(valid_matches)
                        
                        # Update function documentation
                        additional_docs = load_function_documentation(docs_dir, valid_matches)
                        function_docs += "\n\n" + additional_docs
                except Exception as fallback_error:
                    print(f"Fallback parsing also failed: {fallback_error}")
                    log_to_file(trace_log_path, f"FALLBACK PARSING ALSO FAILED: {fallback_error}\n")
            
            # Generate improved kernel with error feedback and memory
            print(f"Generating improved kernel (iteration {iteration + 1})...")
            log_to_file(trace_log_path, f"GENERATING IMPROVED KERNEL (ITERATION {iteration + 1})...")
            
            # Add the error message and documentation to the conversation memory
            error_and_docs_prompt = (
                f"Previous error message:\n"
                f"--------------------------------------------------\n"
                f"{previous_error_message}\n"
                f"--------------------------------------------------\n\n"
                f"Function Documentation:\n"
                f"--------------------------------------------------\n"
                f"{function_docs}\n"
                f"--------------------------------------------------\n\n"
                f"Generate a new improved kernel for this task. Clearly explain your line of reasoning in one sentence, trying "
                f"to keep it as brief as possible. Focus on explaining what solution you are planning on using to "
                f"fix the error. Remember to keep it concise, but explanatory as you will be referencing this later to make sure "
                f"you are not trying to do the same fixes multiple times. "
                f"Your output should include the entire kernel code, NOT just individual fixes. I want to be able to run the code inside the ``` ```"
                f"The way I want your response structured is an explanation of your reasoning at the very start inside *** *** triple stars. "
                f"Then, immediatly after write the python nki code inside triple backticks ``` ```."
                f"I repeat, I only want your output to first be the line of reasoning inside triple stars, then the "
                f"nki kernel code inside triple backticks. Do NOT put the reasoning inside the nki kernel code."
            )
            
            
            # Add user message to memory
            kernel_memory.chat_memory.add_message(HumanMessage(content=error_and_docs_prompt))
            
            # Log the prompt being sent to the LLM
            log_to_file(trace_log_path, f"ERROR REINJECT PROMPT:\n{error_and_docs_prompt}\n")
            
            # Get chat history from memory
            chat_history = kernel_memory.load_memory_variables({})["chat_history"]

            # Create a new message list that explicitly starts with the system message
            # This ensures the system message is always first, regardless of what's in memory
            messages = [SystemMessage(content=system_prompt)]

            # Then add the rest of the messages, but filter out any existing system messages
            # to avoid duplication
            for msg in chat_history:
                if not isinstance(msg, SystemMessage):
                    messages.append(msg)

            # Finally, add the new human message
            messages.append(HumanMessage(content=error_and_docs_prompt))

            # Generate improved response using the properly ordered message list
            improved_generation = kernel_llm.invoke(messages).content
            
            # Add AI response to memory
            kernel_memory.chat_memory.add_message(AIMessage(content=improved_generation))
            
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
                # Also write the reasoning with triple backticks to the output file
                with open(output_address + ".reasoning", "a", encoding="utf-8") as reasoning_file:
                    reasoning_file.write(f"=== Iteration {iteration + 1} ===\n")
                    reasoning_file.write(f"```\n{reasoning_text}\n```")
                    reasoning_file.write("\n\n")
                print("Reasoning extracted and appended to reasoning log.")
                log_to_file(trace_log_path, f"EXTRACTED REASONING:\n{reasoning_text}\n")
                print(reasoning_text)
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
            
            # Now run the test script on the newly generated code
            log_to_file(trace_log_path, f"RUNNING TEST SCRIPT ON UPDATED CODE: {test_script_path}")
            error_message = run_script_and_save_output(test_script_path, test_script_output)
            log_to_file(trace_log_path, f"TEST SCRIPT OUTPUT:\n{error_message}\n")
            
            # Generate a report on the result of the changes
            if iteration > 0:  # Skip for the first iteration as we don't have a previous solution to compare
                print("Generating report on the results of the changes...")
                log_to_file(trace_log_path, "GENERATING REPORT ON RESULTS OF CHANGES...")
                
                # Extract error line from old error message if possible
                old_error_line, _ = extract_error_details(old_error_message)
                new_error_line, _ = extract_error_details(error_message)
                
                old_error_line_info = f"Error occurred at line: {old_error_line}" if old_error_line else "Error line could not be determined."
                new_error_line_info = f"Error occurred at line: {new_error_line}" if new_error_line else "Error line could not be determined."
                
                change_report_prompt = ChatPromptTemplate.from_template(
                    "You are analyzing the results of changes made to fix errors in a NKI kernel.\n\n"
                    "Previous error message:\n{old_error_message}\n\n"
                    "Previous error line information:\n{old_error_line_info}\n\n"
                    "Applied solution (reasoning):\n{reasoning}\n\n"
                    "New error message after applying the solution:\n{new_error_message}\n\n"
                    "New error line information:\n{new_error_line_info}\n\n"
                    "Please provide your analysis in the following JSON format:\n"
                    "```json\n"
                    "{{\n"
                    " \"correct\": boolean, // true if the fix resolved the initial problem, false otherwise\n"
                    " \"report\": \"string\" // brief explanation of why the solution worked or didn't work\n"
                    "}}\n"
                    "```\n\n"
                    "The 'correct' field should be true if the exact error we had last time has been fixed."
                    "it is still deemed correct even if a different error arises, we are just focusing on the "
                    "last error we were trying to fix\n"
                    "Remember, if the previous error and the new error are different, that means the solution is correct and should be true"
                    "Keep your report brief and focused on the specific changes and their effects. This is important"
                    "remember to keep the report consise and focused on key words on why it worked or failed"
                )
                change_report_chain = (
                    change_report_prompt
                    | query_llm
                    | StrOutputParser()
                )
                change_report_json = change_report_chain.invoke({
                    "old_error_message": old_error_message,
                    "old_error_line_info": old_error_line_info,
                    "reasoning": reasoning_text,
                    "new_error_message": error_message,
                    "new_error_line_info": new_error_line_info
                })
                
                # Extract JSON from the response (in case there's additional text)
                json_match = re.search(r'```json\s*(.*?)\s*```', change_report_json, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = change_report_json
                
                # Clean up potential comment lines from the JSON
                json_str = re.sub(r'//.*', '', json_str)
                
                try:
                    report_data = json.loads(json_str)
                    correct = report_data.get("correct", False)
                    report = report_data.get("report", "No explanation provided")
                except json.JSONDecodeError:
                    # Fallback in case JSON parsing fails
                    print("Failed to parse JSON response. Using default values.")
                    correct = False
                    report = change_report_json
                
                # Save the full report (both JSON and extracted values)
                with open(output_address + ".change_reports", "a", encoding="utf-8") as report_file:
                    report_file.write(f"=== Change Report for Iteration {iteration + 1} ===\n")
                    report_file.write(f"Raw response:\n{change_report_json}\n\n")
                    report_file.write(f"Extracted values:\n")
                    report_file.write(f"correct: {correct}\n")
                    report_file.write(f"report: {report}\n")
                    report_file.write("\n\n")
                
                # Also print the report to console
                print(f"\n=== Change Report for Iteration {iteration + 1} ===")
                print(f"correct: {correct}")
                print(f"report: {report}")
                print("\n")
                
                # Log the report
                log_to_file(trace_log_path, f"CHANGE REPORT:\ncorrect: {correct}\nreport: {report}\n")
                
                # Add the report to memory as a system message
                report_message = f"Change Report for Iteration {iteration + 1}: correct={correct}, report={report}"
                kernel_memory.chat_memory.add_message(SystemMessage(content=report_message))
                
                # Update the previous error message for the next iteration
                previous_error_message = error_message
                
                # If no errors, we're done
                if "Error" not in error_message and "error" not in error_message and "ERROR" not in error_message:
                    print("No errors detected! Kernel generation successful.")
                    log_to_file(trace_log_path, "NO ERRORS DETECTED. KERNEL GENERATION SUCCESSFUL.")
                    break
                
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






















if __name__ == "__main__":
    # Define constant file paths
    #TODO change depending on system
    system_prompt_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_prompts/system_prompt_langchain.txt"
    user_prompt_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_prompts/user_prompt_langchain.txt"
    output_address = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/vector_add.txt"  # Raw OpenAI output
    kernel_module_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/vector_add_kernel.py"  # Kernel module file
    test_script_path = "/home/ubuntu/torch2nki/evaluation/samples/test_vector_add.py"
    test_script_output = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/error_message.txt"
    reasoning_log_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/reasoning_log.txt"
    
    # Add path to error documentation
    error_doc_path = "/home/ubuntu/torch2nki/documentation/nki_documentation/nki_error_messages.txt"
    # Add path to function documentation directory
    docs_dir = "/home/ubuntu/torch2nki/documentation/nki_documentation/nki_language_apis_parsed"
    
    # Get credentials
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')

    
    # Run the updated generator with direct documentation and error loop
    generate_kernel_with_direct_docs_and_error_loop(
        system_prompt_path,
        user_prompt_path,
        output_address,
        kernel_module_path,
        test_script_path,
        test_script_output,
        reasoning_log_path,
        error_doc_path,
        docs_dir,
        max_iterations=15
    )