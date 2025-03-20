from openai import OpenAI

def get_chat_completion(system_prompt, user_prompt, model = "gpt-4o-mini", temperature = 0.7):
    """
    Returns the completion for the given system prompt and user prompt.
    
    Args:
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        model (str): The model to use. Defaults to "gpt-4o-mini".
        temperature (float): The temperature to use. Defaults to 0.7.
    
    Returns:
        str: The completion.
    """
    #Set up the client
    client = OpenAI()
    #Perform the completion
    completion = client.chat.completions.create(
        model=model,
        messages=[
            #System prompt
            {"role": "system", "content": system_prompt},
            #User prompt
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature
    )
    #Return the content generated
    return completion.choices[0].message.content

def run_o4(system_prompt_adress, user_prompt_adress, output_adress):
    """
    Runs the OpenAI API to generate a completion from a system prompt and a user prompt.
    
    Args:
        system_prompt_adress (str): The address of the system prompt.
        user_prompt_adress (str): The address of the user prompt.
        output_adress (str): The address of the output.
    """
    #Access system and user prompts from text files in the prompts folder
    with open(system_prompt_adress, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_adress, "r") as f:
        user_prompt = f.read()
    #Get the completion
    result = get_chat_completion(system_prompt, user_prompt)
    #Save the completion as a txt file
    with open(output_adress, "w") as f:
        f.write(result)


if __name__ == "__main__":

    #Run the o4 generator
    run_o4("/home/ubuntu/torch2nki/prompts/system_prompt_naive.txt", "/home/ubuntu/torch2nki/prompts/user_prompt_add.txt", "/home/ubuntu/torch2nki/generation/samples/vector_add.txt")
