from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def copy_main_section_text(url, section_selector="section#nki-language-add", output_file="main_section.txt"):
    # 1. Start a browser (here, Chrome; for Firefox you'd use GeckoDriver)
    driver = webdriver.Chrome()  # If chromedriver isn't in PATH, specify executable_path=...
    
    try:
        # 2. Navigate to the webpage
        driver.get(url)

        # 3. Wait briefly for page content to load (adjust time or use explicit waits)
        time.sleep(2)

        # 4. Locate the main section. 
        #    If you want the entire page, you could use: driver.find_element(By.TAG_NAME, "body")
        #    Otherwise, use a CSS selector like 'section#nki-language-add' or whatever encloses the main content.
        section_element = driver.find_element(By.CSS_SELECTOR, section_selector)

        # 5. Get the *rendered* text of that section as the browser sees it
        section_text = section_element.text

        # 6. Write to a file
        reformatted_text = reformat_nki_doc(section_text)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(reformatted_text)

        print(f"Text copied from '{section_selector}' has been saved to {output_file}")

    finally:
        # 7. Close the browser
        driver.quit()

def reformat_nki_doc(docstring: str) -> str:
    """
    Reformat the given NKI docstring according to the specifications:
      1. Add a new line after the first line (the function name).
      2. Add a line that says "Signature:" before the lines containing the function signature.
      3. Add a new line after the function signature.
      4. Replace "[source]" with "Description:".
      5. Add a new line before "Parameters" and merge it with the next line if it's ":" → "Parameters:".
      6. Add a new line before "Returns" and merge it with the next line if it's ":" → "Returns:".
    """
    lines = docstring.splitlines()
    
    if len(lines) < 3:
        # Not enough lines to reformat reliably, return as-is
        return docstring
    
    # 1) Add a blank line after the first line
    new_lines = ['-----', lines[0], '']  # first line, then a blank line
    
    # 2) Insert "Signature:" and then the second line (function signature), then a blank line
    new_lines.append("Signature:")
    new_lines.append(lines[1])
    new_lines.append('')
    
    # Now process the remaining lines starting from index 2
    i = 2
    while i < len(lines) - 1:
        line_stripped = lines[i].strip()
        
        # 4) Replace "[source]" with "Description:"
        if line_stripped == '[source]':
            new_lines.append("Description:")
            i += 1
            continue
        
        # 5) When encountering "Parameters", add a blank line before it, and then merge with ":" if next line is ":"
        if line_stripped == 'Parameters':
            new_lines.append('')             # new line before "Parameters"
            
            # Check if next line is just ":"
            if i + 1 < len(lines) and lines[i+1].strip() == ':':
                new_lines.append("Parameters:")
                i += 2  # skip the colon line
            else:
                # If for some reason next line isn't ":", just add "Parameters" with no colon
                new_lines.append("Parameters:")
                i += 1
            continue
        
        # 6) Similarly, when encountering "Returns", add a blank line before it, merge with ":" if next line is ":"
        if line_stripped == 'Returns':
            new_lines.append('')
            
            if i + 1 < len(lines) and lines[i+1].strip() == ':':
                new_lines.append("Returns:")
                i += 2
            else:
                new_lines.append("Returns:")
                i += 1
            continue
        
        # Otherwise, just copy the line as-is
        new_lines.append(lines[i])
        i += 1
    
    # Join everything back together
    new_lines.append("")
    return "\n".join(new_lines)

if __name__ == "__main__":
    # LIST THE OPERATIONS IN op_list.py TO PARSE THE CORRESPONDING DOCUMENTATION
    with open("op_list.txt", "r", encoding="utf-8") as file:
        operations = file.readlines()
    
    operations = [line.rstrip("\n") for line in operations]
    print(operations)

    output_file = "nki_documentation/nki_language_apis/nki_data_types.txt"
    for op in operations:
        url_to_parse = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language."+ op + ".html#nki.language." + op
        copy_main_section_text(url_to_parse, section_selector="section#nki-language-" + op.replace('_', '-'), output_file=output_file)