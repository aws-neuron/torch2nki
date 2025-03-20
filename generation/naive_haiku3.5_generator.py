import boto3
import json

def get_chat_completion(system_prompt, user_prompt, model="anthropic.claude-v2", temperature=0.7):
    """
    Returns the completion for the given system prompt and user prompt using AWS Bedrock.
    
    Args:
    system_prompt (str): The system prompt.
    user_prompt (str): The user prompt.
    model (str): The model to use. Defaults to "anthropic.claude-v2".
    temperature (float): The temperature to use. Defaults to 0.7.
    
    Returns:
    str: The completion.
    """
    # Set up the Bedrock client
    bedrock = boto3.client('bedrock-runtime')
    
    # Construct the prompt for Claude-style models
    full_prompt = f"System: {system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"
    
    # Prepare the request body
    body = json.dumps({
        "prompt": full_prompt,
        "max_tokens_to_sample": 300,
        "temperature": temperature,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    })
    
    # Invoke the model
    try:
        response = bedrock.invoke_model(
            modelId=model,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        return response_body['completion']
    
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
        system_prompt = f.read()
    
    with open(user_prompt_address, "r") as f:
        user_prompt = f.read()
    
    # Get the completion
    result = get_chat_completion(system_prompt, user_prompt)
    
    # Save the completion as a txt file
    if result:
        with open(output_address, "w") as f:
            f.write(result)
    else:
        print("Failed to generate completion")

# Example usage
if __name__ == "__main__":
    # Run the Bedrock generation
    run_bedrock_generation(
        "/home/ubuntu/torch2nki/prompts/system_prompt_naive.txt",
        "/home/ubuntu/torch2nki/prompts/user_prompt_add.txt", 
        "/home/ubuntu/torch2nki/generation/samples_bedrock/vector_add_haiku.txt"
    )

# Optional: More flexible model selection
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