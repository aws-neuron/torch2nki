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


if __name__ == "__main__":
    #Access system and user prompts from text files in the prompts folder
    with open("../prompts/system_prompt_naive.txt", "r") as f:
        system_prompt = f.read()
    with open("../prompts/user_prompt_add.txt", "r") as f:
        user_prompt = f.read()
    result = get_chat_completion(system_prompt, user_prompt)
    #save result as a txt file
    with open("../generation/samples/vector_add.txt", "w") as f:
        f.write(result)