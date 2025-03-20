from transformers import GPT2Tokenizer
import re
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
    def __init__(self, tokenizer, config: GenerationConfig):
        self.tokenizer = tokenizer
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
        # Simulated search response, returning "Hello World"
        print(f"Simulated search query: {query}")
        
        simulated_response = "Hello World"
        
        return self._passages2string(simulated_response)

    def _passages2string(self, retrieval_result):
        # Directly return the simulated response (no processing required)
        return retrieval_result

# Example configuration and setup
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GenerationConfig(
    max_turns=10,
    max_start_length=50,
    max_prompt_length=200,
    max_response_length=100,
    max_obs_length=50,
    num_gpus=1,  # or however many GPUs you want to use
    search_url="https://example.com/search",  # Placeholder URL
    topk=3
)

manager = LLMGenerationManager(tokenizer, config)

# Example predictions
predictions = ["<search>How to bake a cake?</search>", "<answer>It is simple to bake a cake.</answer>"]
next_obs, dones = manager.execute_predictions(predictions, pad_token=tokenizer.pad_token, do_search=True)

print(next_obs)
print(dones)
