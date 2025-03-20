import subprocess

def run_script_and_save_output(script_path, output_file):
    # Run the script and capture both stdout and stderr
    result = subprocess.run(
        ['python', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # to get the output as a string instead of bytes
    )
    
    # Combine stdout and stderr
    combined_output = result.stdout + "\n" + result.stderr
    
    # Write the combined output to the specified file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_output)
    
    print(f"Output saved to {output_file}")

if __name__ == '__main__':
    # Change the paths as needed
    script_path = "/home/ubuntu/torch2nki/evaluation/samples/test_vector_add.py"
    output_file = "script_output.txt"
    run_script_and_save_output(script_path, output_file)
