"""
LLM management for the NKI kernel generation pipeline.
Handles direct Bedrock API calls and retry logic.
"""
import boto3
import json
import time
import traceback
from botocore.config import Config
from typing import Optional

from .config import config


class LLMManager:
    """Manages LLM interactions for kernel generation."""
    
    def __init__(self):
        """Initialize the LLM manager."""
        self._bedrock_client = None
    
    def _get_bedrock_client(self):
        """Get or create the Bedrock client."""
        if self._bedrock_client is None:
            boto_config = Config(
                region_name=config.llm_region,
                retries=dict(
                    max_attempts=60,
                    mode="adaptive",
                    total_max_attempts=60
                )
            )
            
            self._bedrock_client = boto3.client(
                'bedrock-runtime',
                config=boto_config
            )
        
        return self._bedrock_client
    
    def call_bedrock_api(self, prompt_text: str, temperature: Optional[float] = None) -> str:
        """Call Claude 3.7 Sonnet via Amazon Bedrock API."""
        if temperature is None:
            temperature = config.llm_temperature
            
        try:
            bedrock = self._get_bedrock_client()
            
            # Prepare the request payload
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config.llm_max_tokens,
                "temperature": temperature,
                "top_p": config.llm_top_p,
                "top_k": config.llm_top_k,
                "stop_sequences": [],
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
                modelId=config.llm_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Process the response
            response_body = json.loads(response.get('body').read())
            
            # Extract the text content from the response
            if "content" in response_body and len(response_body["content"]) > 0:
                for content_item in response_body["content"]:
                    if content_item.get("type") == "text":
                        return content_item.get("text", "")
            
            return ""
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            traceback.print_exc()
            return f"Error occurred: {str(e)}"
    
    def invoke_with_retry(self, prompt_text: str, temperature: Optional[float] = None, 
                         max_retries: Optional[int] = None, initial_backoff: Optional[float] = None) -> str:
        """Invoke the Bedrock API with retry logic."""
        if max_retries is None:
            max_retries = 5
        if initial_backoff is None:
            initial_backoff = config.initial_backoff
            
        for attempt in range(max_retries):
            try:
                return self.call_bedrock_api(prompt_text, temperature)
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff_time = initial_backoff * (2 ** attempt)  # Exponential backoff
                    print(f"Attempt {attempt+1} failed with error: {e}. Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    print(f"All {max_retries} attempts failed. Last error: {e}")
                    raise


# Global LLM manager instance
llm_manager = LLMManager() 