import sys

def copy_alternate_lines(input_file, output_file):
    """
    Copies every other line from input_file to output_file.
    The first line (index 0) is copied, the second line (index 1) is skipped, etc.
    """
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for index, line in enumerate(fin):
            if index % 4 == 0:
                fout.write(line)

if __name__ == "__main__":
    
    input_path = "raw_op_list.txt"
    output_path = "op_list.txt"
    
    copy_alternate_lines(input_path, output_path)
    print(f"Every other line from '{input_path}' has been copied to '{output_path}'.")
