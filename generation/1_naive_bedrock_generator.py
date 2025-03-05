import boto3
import json



def generate(prompt, max_tokens = 500, temperature = 0.7, top_k = 50, top_p = 0.9):
    """
    Generates a text completion using the specified model from AWS Bedrock.

    Parameters:
    - prompt (str): The initial text prompt to generate the completion from.
    - max_tokens (int, optional): The maximum number of tokens to generate. Default is 500.
    - temperature (float, optional): Controls the randomness of predictions. Default is 0.7.
    - top_k (int, optional): The number of highest probability vocabulary tokens to keep for sampling. Default is 50.
    - top_p (float, optional): The cumulative probability of parameter highest probability tokens. Default is 0.9.

    Returns:
    - str: The generated text completion.
    """
    #Make the AWS Bedrock Client

    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

    #Define the payload
    payload = {
    "prompt": prompt, 
    "max_tokens_to_sample": max_tokens,
    "temperature": temperature, 
    "top_k": top_k,
    "top_p": top_p,
    }
    payload_str = json.dumps(payload)

    #Call the model
    response = bedrock_runtime.invoke_model(
    modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=payload_str
    )

    #parse & return
    response_body_text = json.loads(response["body"].read().decode("utf-8"))["completion"]
    return response_body_text



def main():
    print(generate("Hello, how are you?"))



if __name__ == "__main__":
    main()
