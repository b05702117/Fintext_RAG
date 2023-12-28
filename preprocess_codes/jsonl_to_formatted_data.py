import json
import os
import argparse
from pathlib import Path
from transformers import BertTokenizer

ROOT = "/home/ybtu/FinNLP"
SOURCE_DIR = "/home/ybtu/FinNLP/Jsonl_Data"
DEST_DIR = "/home/ybtu/FinNLP/collections"

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
            "name": first_line["name"],
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

    for file_name in os.listdir(SOURCE_DIR):
        converted_data = convert_jsonl_format(os.path.join(SOURCE_DIR, file_name), format_type)
        save_data_to_jsonl(converted_data, os.path.join(dest_dir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL data to desired format")
    parser.add_argument("format_type", choices=["basic", "multi_fields", "meta_data"], help="Choose the format type to convert the JSONL data")
    args = parser.parse_args()

    main(args.format_type)


