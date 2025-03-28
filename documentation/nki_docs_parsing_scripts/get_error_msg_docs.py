import requests
from bs4 import BeautifulSoup
import os

def extract_nki_errors(url, output_path):
    # Fetch the webpage
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare to write errors to a text file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Find all error-related sections
        error_sections = soup.find_all('section', id=lambda x: x and x.startswith('err-'))
        
        for section in error_sections:
            # Extract error name (from the section id)
            error_name = section.get('id', 'Unknown Error').replace('err-', '')
            
            # Write error name
            outfile.write(f"ERROR: {error_name}\n")
            outfile.write("="*50 + "\n")
            
            # Gather all paragraphs and code blocks
            content_elements = []
            
            # Add paragraphs
            paragraphs = section.find_all('p')
            content_elements.extend(paragraphs)
            
            # Find and extract code blocks from both docutils and highlight classes
            code_blocks = section.find_all(['code', 'div'], class_=['docutils literal notranslate', 'highlight-python notranslate'])
            content_elements.extend(code_blocks)
            
            # Keep track of instruction and code example counts
            instruction_count = 0
            code_example_count = 0
            
            # Process content elements in order
            for element in content_elements:
                # Handle paragraph elements (instructions)
                if element.name == 'p':
                    para_text = element.get_text(strip=True)
                    if para_text:
                        instruction_count += 1
                        outfile.write(f"Instruction {instruction_count}: {para_text}\n")
                
                # Handle code blocks
                elif element.name in ['code', 'div']:
                    # Extract text differently based on tag type
                    if element.name == 'code':
                        code_text = element.get_text(strip=True)
                    else:
                        # For highlight divs, look for code within spans
                        code_text = ' '.join([span.get_text(strip=True) for span in element.find_all('span')])
                    
                    if code_text:
                        code_example_count += 1
                        outfile.write(f"Code Example {code_example_count}:\n{code_text}\n")
            
            # Add a separator between errors
            outfile.write("\n" + "="*60 + "\n\n")
    
    print(f"Errors have been saved to {output_path}")

# URL of the NKI API Errors documentation
url = 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.errors.html'

# Specific output path you provided
output_path = os.path.join(os.path.dirname(__file__), '..', 'nki_documentation', 'nki_error_messages.txt')

# Run the scraper
extract_nki_errors(url, output_path)