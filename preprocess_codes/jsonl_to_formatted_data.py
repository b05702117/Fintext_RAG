import json
import os
import argparse
import glob
from pathlib import Path
import re
import spacy
from transformers import BertTokenizer

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

def convert_docid_to_title(docid):
    ''' TODO: wait for the mapping of part_key and item_key from YT'''

    with open('cik_to_company.json', 'r') as f:
        cik_to_company = json.load(f)

    with open('item_mapping.json', 'r') as f:
        item_mapping = json.load(f)

    # 20220125_10-Q_789019_part1_item2_para475
    components = docid.split('_')
    
    # Extract relevant parts
    date_str, form, cik = components[0], components[1], components[2]
    part, item = None, None
    if len(components) == 6:
        part, item = components[3], components[4]
    elif len(components) == 5:
        # 20211216_10-K_315189_mda_para398
        item = components[3]
    else:
        print("Invalid docid:", docid)
        return None

    # Convert date to year and quarter (assuming the date format is yyyymmdd)
    year = date_str[:4]
    month = int(date_str[4:6])
    quarter = (month - 1) // 3 + 1

    # Get the company name from CIK
    company_name = cik_to_company.get(cik, "Unknown Company")

    # Get the item name from item number
    item_name = item_mapping.get(item, "Unknown Item")

    # Formant the new title
    new_title = f"{company_name} {year} Q{quarter} {form} {item_name}"

    return new_title

def convert_data(first_line, data, format_type):
    # Regular expression pattern to match ids ending with "_fin_stats_para" followed by numbers
    pattern = r'_finstats_para\d+$'
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

    elif format_type == "meta_data":
        meta_data = f"cik: {first_line['cik']}, company_name: {first_line['company_name']}, filing_date: {first_line['filing_date']}, form: {first_line['form']}"
        base_data["contents"] = meta_data + "; " + base_data["contents"]

    elif format_type == "title":
        title = convert_docid_to_title(data["id"])
        if title:
            base_data["contents"] = title + "; " + base_data["contents"]
        else:
            meta_data = f"cik: {first_line['cik']}, company_name: {first_line['company_name']}, filing_date: {first_line['filing_date']}, form: {first_line['form']}"
            base_data["contents"] = meta_data + "; " + base_data["contents"]
    
    elif format_type == "ner":
        # Add NER tags to the data
        base_data["NER"] = dict()
        
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(base_data["contents"])
        for ent in doc.ents:
            if ent.label_ not in base_data["NER"]:
                base_data["NER"][ent.label_] = []
            base_data["NER"][ent.label_].append(ent.text)
    
    elif format_type == "ner_concat":
        # Concatenate NER tags to the data
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(base_data["contents"])

        modified_contents = []
        last_end = 0
        for ent in doc.ents:
            # Add the text before the entity
            modified_contents.append(base_data["contents"][last_end:ent.start_char])
            # Add the entity with its label
            modified_contents.append(f"[{ent.label_}: {ent.text}]")
            # Update the last_end position
            last_end = ent.end_char

        # Add the remaining text after the last entity
        modified_contents.append(base_data["contents"][last_end:])

        # Concatenate the parts to form the complete text
        base_data["contents"] = "".join(modified_contents)

        # Prepend title to the contents
        title = convert_docid_to_title(data["id"])
        if title:
            base_data["contents"] = title + "; " + base_data["contents"]
        else:
            meta_data = f"cik: {first_line['cik']}, company_name: {first_line['company_name']}, filing_date: {first_line['filing_date']}, form: {first_line['form']}"
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

def main(format_type, start_index, end_index):
    dest_dir = os.path.join(DEST_DIR, format_type)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    file_paths = visit_jsonl_files_under_dir(SOURCE_DIR)
    total_files = len(file_paths)
    print(f"Converting {total_files} files to {format_type} format")
    print(f"Start index: {start_index}, End index: {end_index}")
    
    # for idx, file_path in enumerate(file_paths):
    for idx in range(start_index, min(end_index, total_files)):
        file_path = file_paths[idx]
        converted_data = convert_jsonl_format(file_path, format_type)
        file_name = Path(file_path).name
        save_data_to_jsonl(converted_data, os.path.join(dest_dir, file_name))

        # Calculate the progress
        progress = (idx + 1) / total_files * 100

        # Check if progress has reached a multiple of 10%
        if (idx + 1) % max(1, total_files // 10) == 0 or (idx + 1) == total_files:
            print(f"{progress:.0f}% complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL data to desired format")
    parser.add_argument("format_type", choices=["basic", "multi_fields", "meta_data", "title", "ner", "ner_concat"], help="Choose the format type to convert the JSONL data")
    parser.add_argument("start_index", type=int, help="The starting index of files to process")
    parser.add_argument("end_index", type=int, help="The ending index of files to process")
    args = parser.parse_args()

    main(args.format_type, args.start_index, args.end_index)


