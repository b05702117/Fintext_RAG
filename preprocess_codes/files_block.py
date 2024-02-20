import os
import argparse
from pathlib import Path

import glob

ROOT = "/home/ybtu/FinNLP"
# SOURCE_DIR = "/home/ybtu/FinNLP/Jsonl_Data"
SOURCE_DIR = "/home/ythsiao/output"
DEST_DIR = "/home/ybtu/FinNLP/collections/10K"
# DEST_DIR = "/tmp2/ybtu/FinNLP/collections"

def visit_jsonl_files_under_dir(root_dir):
    ''' Visit all .jsonl files under the root directory and return a list of their paths '''
    # Pattern to match all .jsonl files under the root directory
    pattern = os.path.join(root_dir, '**', '*.jsonl')
    
    # List to store the paths of the files
    jsonl_files = []

    # Walk through the directory and find all .jsonl files
    for filename in glob.glob(pattern, recursive=True):
        jsonl_files.append(filename)
        # Here you can add code to "visit" each file, e.g., read them, print their names, etc.

    # Sort the list of files before returning
    jsonl_files.sort()

    return jsonl_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL data to desired format")
    parser.add_argument("format_type", choices=["basic", "multi_fields", "meta_data", "title", "ner", "ner_concat"], help="Choose the format type to convert the JSONL data")
    args = parser.parse_args()

    file_paths = visit_jsonl_files_under_dir(SOURCE_DIR)
    total_files = len(file_paths)
    segment_size = total_files // 5  # 20% of total files

    for i in range(5):
        start_index = i * segment_size
        end_index = start_index + segment_size

        # Adjust end_index for the last segment to include any remaining files
        if i == 4:
            end_index = total_files

        print(f"tmux new-session -d -s session_{i} 'python script_name.py {args.format_type} {start_index} {end_index}'")
