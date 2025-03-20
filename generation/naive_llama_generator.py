import boto3
import json

def get_chat_completion(system_prompt, user_prompt, model="meta.llama3-70b-instruct-v1:0", temperature=0.5):
    """
    Returns the completion for the given system prompt and user prompt using AWS Bedrock.
    
    Args:
    system_prompt (str): The system prompt.
    user_prompt (str): The user prompt.
    model (str): The model to use. Defaults to "meta.llama3-70b-instruct-v1:0".
    temperature (float): The temperature to use. Defaults to 0.5.
    
    Returns:
    str: The completion.
    """
    # Set up the Bedrock client
    bedrock = boto3.client('bedrock-runtime')
    
    # Construct the prompt for Llama 3
    # Note: Llama 3 uses a different prompt format compared to Claude
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    
    # Prepare the request body
    body = json.dumps({
        "prompt": full_prompt,
        "max_gen_len": 512,
        "temperature": temperature,
        "top_p": 0.9
    })
    
    # Invoke the model
    try:
        response = bedrock.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Extract the generated text 
        # Note: The exact key might depend on the Bedrock response format for Llama 3
        return response_body.get('generation', '')
    
    except Exception as e:
        print(f"Error in Bedrock API call: {e}")
        return None

def run_bedrock_generation(system_prompt_address, user_prompt_address, output_address):
    """
    Runs the Bedrock API to generate a completion from a system prompt and a user prompt.
    
    Args:
    system_prompt_address (str): The address of the system prompt file.
    user_prompt_address (str): The address of the user prompt file.
    output_address (str): The address to save the output.
    """
    # Read system and user prompts from text files
    with open(system_prompt_address, "r") as f:
        system_prompt = f.read().strip()
    
    with open(user_prompt_address, "r") as f:
        user_prompt = f.read().strip()
    
    # Get the completion
    result = get_chat_completion(system_prompt, user_prompt)
    
    # Save the completion as a txt file
    if result:
        with open(output_address, "w") as f:
            f.write(result)
    else:
        print("Failed to generate completion")

def list_available_models():
    """
    Lists available Bedrock foundation models.
    """
    bedrock = boto3.client('bedrock')
    models = bedrock.list_foundation_models()
    for model in models['modelSummaries']:
        print(f"Model ID: {model['modelId']}")
        print(f"Provider: {model['providerName']}")
        print(f"Input Modalities: {model.get('inputModalities', 'N/A')}")
        print(f"Output Modalities: {model.get('outputModalities', 'N/A')}")
        print("---")

# Example usage
if __name__ == "__main__":
    # Run the Bedrock generation
    run_bedrock_generation(
        "/home/ubuntu/torch2nki/prompts/system_prompt_naive.txt",
        "/home/ubuntu/torch2nki/prompts/user_prompt_add_llama.txt",
        "/home/ubuntu/torch2nki/generation/samples_bedrock/vector_add_llama3.txt"
    )