"""
JSON utility functions for the NKI kernel generation pipeline.
"""
import json
import re
from typing import List, Any


def extract_json_array(text: str) -> str:
    """Clean up text to extract a JSON array."""
    # Remove any non-JSON text before or after the array
    text = text.strip()
    # If text begins with characters before [, remove them
    if '[' in text and text[0] != '[':
        text = text[text.find('['):]
    # If text has characters after the closing ], remove them
    if ']' in text and text[-1] != ']':
        text = text[:text.rfind(']')+1]
    # If we still don't have a valid JSON looking text, try regex
    if not (text.startswith('[') and text.endswith(']')):
        json_pattern = re.compile(r'\[.*?\]', re.DOTALL)
        json_match = json_pattern.search(text)
        if json_match:
            text = json_match.group(0)
    return text


def parse_json_response(response: str, fallback_pattern: str = r'["\']([\w_-]+)["\']') -> List[str]:
    """Parse JSON response with fallback to regex extraction."""
    try:
        # Clean the response and try to parse it
        cleaned_response = extract_json_array(response)
        
        # Handle empty lists represented as empty string, "[]", etc.
        if not cleaned_response or cleaned_response.isspace():
            return []
        elif cleaned_response == "[]":
            return []
        else:
            return json.loads(cleaned_response)
            
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        
        # Fallback mechanism: try to extract using regex
        try:
            pattern = re.compile(fallback_pattern)
            matches = pattern.findall(response)
            print(f"Using fallback: Extracted via regex: {', '.join(matches)}")
            return matches
        except Exception as fallback_error:
            print(f"Fallback parsing also failed: {fallback_error}")
            return []


def extract_json_from_response(response: str) -> str:
    """Extract JSON from a response that might contain additional text."""
    # Try to extract JSON from code blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # If no code block, try to find JSON directly
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # Return the original response if no JSON found
    return response


def clean_json_string(json_str: str) -> str:
    """Clean up a JSON string by removing comments and extra whitespace."""
    # Remove comment lines
    json_str = re.sub(r'//.*', '', json_str)
    # Remove extra whitespace
    json_str = re.sub(r'\s+', ' ', json_str).strip()
    return json_str 