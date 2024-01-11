import json
import os
import argparse
import glob
from pathlib import Path
import re
from transformers import BertTokenizer

ROOT = "/home/ybtu/FinNLP"
# SOURCE_DIR = "/home/ybtu/FinNLP/Jsonl_Data"
SOURCE_DIR = "/home/ythsiao/output"
DEST_DIR = "/home/ybtu/FinNLP/collections"
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

    return jsonl_files

def read_jsonl_file(file_path):
    with open(file_path, "r") as file:
        first_line = json.loads(next(file))
        for line in file:
            yield first_line, json.loads(line)

def filter_paragraphs(paragraphs):
    ''' filter paragraphs with less than 10 tokens '''
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    filtered_paragraphs = []
    for paragraph in paragraphs:
        tokens = tokenizer.tokenize(paragraph)
        if len(tokens) >= 10:
            filtered_paragraphs.append(paragraph)
    return filtered_paragraphs

def convert_data(first_line, data, format_type):
    # Regular expression pattern to match ids ending with "_fin_stats_para" followed by numbers
    pattern = r'_fin_stats_para\d+$'
    if re.search(pattern, data["id"]):
        return None

    if data["id"].endswith("statements") or data["id"].endswith("sheets"):
        return None

    base_data = {
        "id": data["id"],
        "contents": " ".join(data["paragraph"])
    }

    # filter paragraphs
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # num_tokens = len(tokenizer.tokenize(base_data["contents"]))
    # num_tokens = len(base_data["contents"].split(" "))
    # if num_tokens <= 10:
    #     return None
    
    if format_type == "multi_fields":
        base_data.update({
            "cik": first_line["cik"],
            "company_name": first_line["company_name"],
            "filing_year": first_line["filing_date"][:4],
            "filing_month": first_line["filing_date"][4:6],
            "filing_day": first_line["filing_date"][6:],
            "form": first_line["form"],
            "order": data["order"]
        })

    # elif format_type == "meta_data":
    #     base_data["metadata"] = {
    #         "cik": first_line["cik"],
    #         "name": first_line["name"],
    #         "filing_date": first_line["filing_date"],
    #         "form": first_line["form"],
    #         "order": data["order"]
    #     }

    elif format_type == "meta_data":
        meta_data = f"cik: {first_line['cik']}, name: {first_line['name']}, filing_date: {first_line['filing_date']}, form: {first_line['form']}"
        base_data["contents"] = meta_data + "; " + base_data["contents"]

    return base_data

def convert_jsonl_format(file_path, format_type):
    converted_data = []
    for first_line, data in read_jsonl_file(file_path):
        converted = convert_data(first_line, data, format_type)
        if converted:
            converted_data.append(converted)
    return converted_data

def save_data_to_jsonl(data, file_path):
    with open(file_path, "w") as file:
        for line in data:
            json.dump(line, file)
            file.write("\n")

def main(format_type):
    dest_dir = os.path.join(DEST_DIR, format_type)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    file_paths = visit_jsonl_files_under_dir(SOURCE_DIR)
    for file_path in file_paths:
        converted_data = convert_jsonl_format(file_path, format_type)
        file_name = Path(file_path).name
        save_data_to_jsonl(converted_data, os.path.join(dest_dir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL data to desired format")
    parser.add_argument("format_type", choices=["basic", "multi_fields", "meta_data"], help="Choose the format type to convert the JSONL data")
    args = parser.parse_args()

    main(args.format_type)


