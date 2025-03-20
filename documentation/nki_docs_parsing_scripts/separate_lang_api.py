import os
import sys

def split_text_file(file_path, target_directory):
    """Splits the content of the file based on '-----' delimiters and creates new files."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by '-----' to get individual sections
    sections = content.split('-----')

    # Filter out empty sections (strip whitespace and ignore sections that are just empty)
    sections = [section.strip() for section in sections if section.strip()]

    # Create a new file for each section
    for section in sections:
        # Get the first line (function name) from the section
        first_line = section.splitlines()[0].strip()

        # Skip if the first line is blank (even though it shouldn't happen)
        if not first_line:
            continue

        # Create a valid file name by replacing '.' with '_'
        file_name = first_line.replace('.', '_') + '.txt'
        
        # Ensure the target directory exists
        os.makedirs(target_directory, exist_ok=True)
        
        # Create the full path for the new file
        new_file_path = os.path.join(target_directory, file_name)

        # Write the section into a new file
        with open(new_file_path, 'w') as new_file:
            new_file.write(section)

        print(f"Created file: {new_file_path}")

def process_directory(source_directory, target_directory):
    """Process all .txt files in the given directory."""
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        # Only process .txt files (non-recursively)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print(f"Processing file: {file_name}")
            split_text_file(file_path, target_directory)

if __name__ == "__main__":
    # Hardcoded directories
    source_directory = '../nki_documentation/nki_language_apis'
    target_directory = '../nki_documentation/nki_language_apis_parsed'

    # Start processing
    process_directory(source_directory, target_directory)
