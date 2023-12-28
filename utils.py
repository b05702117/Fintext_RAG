import json
import os 
import shutil
import subprocess
import fnmatch
import re
from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR

def aggregate_and_index_all_prior(cik, form, start_year, end_year):
    # Define the index name and path
    index_name = f"up_to_{end_year}_{form}_{cik}"
    index_path = os.path.join(INDEX_DIR, str(cik), index_name)

    # Check if the index already exists
    if os.path.isdir(index_path):
        print(f"Index {index_name} already exists.")
        return index_name

    # Create a temporary directory for aggregation
    tmp_dir = os.path.join(ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        # Find and copy relevant JSON files to the temporary directory
        # search_pattern = f"{year}????_{form}_{cik}.json"
        search_pattern = f"{year}????_{form}_{cik}.jsonl"
        for filename in os.listdir(FORMMATED_DIR):
            if fnmatch.fnmatch(filename, search_pattern):
                shutil.copy(os.path.join(FORMMATED_DIR, filename), tmp_dir)

    # Run Pyserini's indexing command
    subprocess.run([
        "python", "-m", "pyserini.index.lucene", 
        "-collection", "JsonCollection", 
        "-input", tmp_dir, 
        "-index", index_path, 
        "-generator", "DefaultLuceneDocumentGenerator", 
        "-threads", "1", 
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw"
    ])

    # Clean up the temporary directory
    shutil.rmtree(tmp_dir)

    return index_name

def aggregate_and_index(cik, year, form):
    # Create a temporary directory for aggregation
    tmp_dir = os.path.join(ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Find and copy relevant JSON files to the temporary directory
    # search_pattern = f"{year}????_{form}_{cik}.json"
    search_pattern = f"{year}????_{form}_{cik}.jsonl"
    for filename in os.listdir(FORMMATED_DIR):
        if fnmatch.fnmatch(filename, search_pattern):
            shutil.copy(os.path.join(FORMMATED_DIR, filename), tmp_dir)

    # Define the index name and path
    index_name = f"{year}_{form}_{cik}"
    index_path = os.path.join(INDEX_DIR, str(cik), index_name)

    # Run Pyserini's indexing command
    subprocess.run([
        "python", "-m", "pyserini.index.lucene", 
        "-collection", "JsonCollection", 
        "-input", tmp_dir, 
        "-index", index_path, 
        "-generator", "DefaultLuceneDocumentGenerator", 
        "-threads", "1", 
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw"
    ])

    # Clean up the temporary directory
    shutil.rmtree(tmp_dir)

    return index_name

def retrieve_index(cik, form, year, month=None):
    if month is None:
        search_pattern = f"{year}_{form}_{cik}"
    else:
        search_pattern = f"{year}{month:02}??_{form}_{cik}"

    for file_name in os.listdir(os.path.join(INDEX_DIR, cik)):
        if fnmatch.fnmatch(file_name, search_pattern):
            return file_name
    
    print("Index not found.")
    return None

def simple_sentence_splitter(text):
    sentences = text.split(". ")
    sentences = [s.strip() for s in sentences if s]  # Remove any empty strings
    return sentences

def extract_contents_from_hit(hit):
    parsed_json = json.loads(hit.raw) # Convert the raw string into a JSON object
    contents = parsed_json['contents'] # Retrieve the 'contents' value from the JSON object
    return contents

def find_highest_prob_word(data, n):
    words_tgt = data[0]['words_tgt']
    word_probs_tgt = data[0]['word_probs_tgt']

    sorted_indices = word_probs_tgt.argsort()[::-1]  # Sort indices in descending order
    top_n_indices = sorted_indices[:n]  # Get the top-n indices

    top_n_words = [words_tgt[i] for i in top_n_indices]

    return top_n_words 


def get_file_name(cik, form, year, month):
    search_pattern = f"{year}{month:02d}??_{form}_{cik}.jsonl"
    for file_name in os.listdir(FORMMATED_DIR):
        if fnmatch.fnmatch(file_name, search_pattern):
            return file_name
    print("File not found.")
    return None

def retrieve_paragraph_from_raw_jsonl(file_name, part_key, item_key, paragraph_number):
    search_pattern = f"*_{part_key}_{item_key}_para{paragraph_number}"

    with open(os.path.join(RAW_DIR, file_name), "r") as open_file:
        next(open_file) # skip the first line

        for line in open_file:
            data = json.loads(line)

            if not re.search(r'para\d+$', data.get("id", "")):
                continue
            
            if fnmatch.fnmatch(data["id"], search_pattern):
                return data["paragraph"]
            
    print("Paragraph not found.")
    return None

def retrieve_paragraph_from_fromatted_jsonl(file_name, part_key, item_key, paragraph_number):
    search_pattern = f"*_{part_key}_{item_key}_para{paragraph_number}"
    with open(os.path.join(FORMMATED_DIR, file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                return data["contents"]
            
    print("Paragraph not found.")
    return None

def retrieve_paragraph_from_docid(docid):
    file_name = docid.split('_')[0] + '_' + docid.split('_')[1] + '_' + docid.split('_')[2] + '.jsonl'
    with open(os.path.join(FORMMATED_DIR, file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if data["id"] == docid:
                return data["contents"]
    print("Paragraph not found.")
    return None