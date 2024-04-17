import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import json
import os
import argparse
import glob
from pathlib import Path
import re
import spacy
from spacy_llm.util import assemble
from transformers import BertTokenizer
from config import CIK_TO_COMPANY, ITEM_MAPPING, CIK_TO_SECTOR
from utils import convert_docid_to_title, docid_to_cik

SOURCE_DIR = "/home/ybtu/FinNLP/collections/10K/ner_concat"
DEST_DIR = "/home/ybtu/FinNLP/collections/10K"

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

def prepend_info(file_path, prepend_info):
    prepended_data = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            docid = data["id"]
            para_content = data["contents"]

            if prepend_info == "title":
                title = convert_docid_to_title(docid)
                new_content = f"{title}; {para_content}"
            elif prepend_info == "company_name":
                cik = docid_to_cik(docid)
                company_name = CIK_TO_COMPANY.get(cik, "Unknown Company")
                new_content = f"{company_name}; {para_content}"

            # write the new content to the file
            # with open(file_path, "w") as f:
            #     f.write(new_content)

            print(new_content)

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL data to desired format")
    parser.add_argument("--prepend_info", choices=["title", "company_name"])

    args = parser.parse_args()

    dest_dir = os.path.join(DEST_DIR, f"ner_concat_with_{args.prepend_info}")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    source_file_paths = visit_jsonl_files_under_dir(SOURCE_DIR)
    total_files = len(source_file_paths)

    print(f"Converting files from {SOURCE_DIR} to {dest_dir}")
    print(f"Total files: {total_files}")

    for idx, source_file_path in enumerate(source_file_paths):
        file_name = Path(source_file_path).name
        dest_file_path = os.path.join(dest_dir, file_name)

        with open(source_file_path, "r") as source_file, open(dest_file_path, "w") as destination_file:
            for line in source_file:
                data = json.loads(line)
                docid = data["id"]
                para_content = data["contents"]

                if args.prepend_info == "title":
                    title = convert_docid_to_title(docid)
                    new_content = f"{title}; {para_content}"
                elif args.prepend_info == "company_name":
                    cik = docid_to_cik(docid)
                    company_name = CIK_TO_COMPANY.get(cik, "Unknown Company")
                    new_content = f"{company_name}; {para_content}"

                # write the new content to the file
                destination_file.write(json.dumps({"id": docid, "contents": new_content}) + "\n")

if __name__ == "__main__":
    main()