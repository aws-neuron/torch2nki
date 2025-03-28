from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    search_url: str
    topk: int = 3

class LLMGenerationManager:
    def __init__(self, tokenizer, model, config: GenerationConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

    def _tokenize_single(self, response: str):
        return self.tokenizer(response, add_special_tokens=False, return_tensors='pt')['input_ids']

    def _postprocess_responses(self, responses):
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        responses_str = [
            resp.split('</search>')[0] + '</search>' if '</search>' in resp else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp else resp
            for resp in responses_str
        ]
        responses = self._tokenize_single(responses_str[0])
        return responses, responses_str

    def _process_next_obs(self, next_obs: str):
        return self.tokenizer(next_obs, padding='longest', return_tensors='pt', add_special_tokens=False)['input_ids']

    def execute_predictions(self, predictions, pad_token, do_search=True):
        actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        search_results = []

        # Process search queries one by one
        for query in contents:
            if query:  # Only perform search if there's a query
                search_results.append(self._single_search(query))
            else:
                search_results.append('')

        for action, search_result in zip(actions, search_results):
            if action == 'search':
                next_obs.append(f'\n\n<information>{search_result.strip()}</information>\n\n')
                dones.append(0)
            elif action == 'answer':
                next_obs.append('')
                dones.append(1)
            else:
                next_obs.append(f'\nInvalid action. Please use <search> or <answer>.</n>')
                dones.append(0)

        return next_obs, dones

    def postprocess_predictions(self, predictions):
        actions, contents = [], []
        for prediction in predictions:
            if isinstance(prediction, str):
                match = re.search(r'<(search|answer)>(.*?)</\1>', prediction, re.DOTALL)
                if match:
                    actions.append(match.group(1))
                    contents.append(match.group(2).strip())
                else:
                    actions.append(None)
                    contents.append('')
        return actions, contents

    def _single_search(self, query):
        # Always return "Hello World" as a dummy response for any search query
        print(f"Simulated search query: {query}")
        
        simulated_response = "Hello World"
        
        return self._passages2string(simulated_response)

    def _passages2string(self, retrieval_result):
        # Directly return the simulated response (no processing required)
        return retrieval_result

    def generate_response(self, question: str):
        # Tokenize the input question
        input_ids = self.tokenizer.encode(question, return_tensors="pt")

        # Set `pad_token_id` and `attention_mask` properly
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create attention mask (1 for real tokens, 0 for padding)
        self.model.config.pad_token_id = self.model.config.eos_token_id  # Use EOS token as padding token for GPT-2

        # Generate a response from the model
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  # Set the attention mask
            max_length=self.config.max_response_length,
            num_return_sequences=1,
            temperature=0.7,  # Increase temperature for more diverse results
            top_k=50,  # Top-k sampling for more variety
            top_p=0.95  # Nucleus sampling (top-p sampling) for more diversity
        )

        # Decode and process the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response


# Example configuration and setup
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Load the GPT-2 model
config = GenerationConfig(
    max_turns=10,
    max_start_length=50,
    max_prompt_length=600,
    max_response_length=1024,
    max_obs_length=50,
    num_gpus=1,  # or however many GPUs you want to use
    search_url="https://example.com/search",  # Placeholder URL
    topk=3
)

manager = LLMGenerationManager(tokenizer, model, config)

# Directly input the question here
question = """Answer the given question: What is 1+1? You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n Here is an example: <think>I need to search for Paul Walker's cars in Fast and Furious movies.</think>
<search> Paul Walker's cars in Fast and Furious </search>

<information>Doc 1(Title: "Paul Walker") also had a large car collection of about 30 cars, a portion of which he co-owned along with Roger Rodas. The collection included (but is not limited to): Roger Rodas became Walker's financial adviser in 2007 and helped to establish Reach Out Worldwide. Rodas, a pro-am racer was the CEO of Always Evolving, a Valencia high-end vehicle performance shop owned by Walker. Walker was also a close friend of his ""2 Fast 2 Furious"" co-star Tyrese Gibson. Vin Diesel considered Walker to be like a brother, both on and off screen, and affectionately called him ""Pablo"". Walker's mother referred to
Doc 2(Title: "Paul Walker") Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and ""Varsity Blues"". In 2001, he gained international fame for his role in the street racing action film ""The Fast and the Furious"" (2001), a role he reprised in five of the next six installments, but died in 2013 in the middle of filming ""Furious 7"" (2015). Walker began his career guest-starring on
Doc 3(Title: "Paul Walker") of Porsche in a separate lawsuit filed by Roger Rodas' widow, Kristine. The ruling had no bearing on two other cases against Porsche which have been filed by Walker's father, who is also the executor of the actor's estate, and his daughter. Walker's father and daughter both reached an agreement with Porsche. Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and</information>

From the information provided, it's clear that Paul Walker was a part of the "Fast and Furious" series, but the specific list of cars is not mentioned. Since I lack this particular detail, I will call a search engine to get the specific list of cars Paul Walker drove in the "Fast and Furious" movies.

<answer> Charger </answer>\n

 Now recall, you must answer the question: is our question: What is 1+1?"""

# Generate and print the response
response = manager.generate_response(question)
print("Generated Response:", response)
