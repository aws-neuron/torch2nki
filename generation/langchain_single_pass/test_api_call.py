import json
import boto3
import sys

def read_prompt_from_file(file_path):
    """Read prompt content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def call_claude_api(prompt_text):
    """Call Claude 3.7 Sonnet via Amazon Bedrock API."""
    try:
        # Initialize the Bedrock Runtime client
        bedrock = boto3.client('bedrock-runtime')
        
        # Prepare the request payload
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
        }
        
        # Make the API call
        response = bedrock.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # Process and return the response
        response_body = json.loads(response.get('body').read())
        return response_body
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        sys.exit(1)

def main():
    # File path for the prompt
    file_path = "/home/ubuntu/torch2nki/prompts/ctc_nki_prompt.txt"
    
    # Read prompt from file
    prompt_text = read_prompt_from_file(file_path)
    print(f"Prompt from file: {prompt_text}\n")
    
    # Call the API
    print("Calling Claude 3.7 Sonnet API...")
    response = call_claude_api(prompt_text)
    
    # Print formatted response
    print("\nAPI Response:")
    print(json.dumps(response, indent=2))
    
    # Extract and print just the model's response text for clarity
    if "content" in response and len(response["content"]) > 0:
        for content_item in response["content"]:
            if content_item.get("type") == "text":
                print("\nClaude's response text:")
                print(content_item.get("text"))

if __name__ == "__main__":
    main()