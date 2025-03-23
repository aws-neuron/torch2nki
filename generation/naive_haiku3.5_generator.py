import boto3
import json

def get_chat_completion(system_prompt, user_prompt, model="anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.7):
    """
    Returns the completion for the given system prompt and user prompt using AWS Bedrock with Messages API.
    """
    # Set up the Bedrock client
    bedrock = boto3.client('bedrock-runtime')
    
    # Construct the request body for Messages API
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    })
    
    # Invoke the model
    try:
        response = bedrock.invoke_model(
            modelId=model,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
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