import sys
sys.path.append("/home/ybtu/FinNLP")

from config import FORMMATED_DIR
from utils import retrieve_paragraph_from_docid, get_10K_file_name

import os
import random
import json
import fnmatch
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from transformers import BertTokenizer

random.seed(52)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=2000)
parser.add_argument("--sample_item", type=str, default="item7")
parser.add_argument("--start_year", type=int, default=2018)
parser.add_argument("--sample_year", type=int, default=2021)
parser.add_argument("--positive_number", type=int, default=5)
parser.add_argument("--negative_number", type=int, default=5)

args = parser.parse_args()

def tfidf_similarity(documents: list[str], query: str) -> np.ndarray:
    """
    Calculate the TF-IDF cosine similarity between a query and a set of documents.

    Parameters:
    documents (List[str]): A list of documents.
    query (str): The query string.

    Returns:
    np.ndarray: An array of cosine similarity scores between the query and each document.
    """
    # Combine the query with the document corpus
    combined = documents + [query]

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents and the query
    tfidf_matrix = vectorizer.fit_transform(combined)

    # Calculate cosine similarity between the query and the documents
    # Query is the last vector in the matrix
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Flatten to convert 2D array to 1D array
    return cosine_similarities.flatten()

# Function to sample 2000 paragraph ids
def sample_paragraph_ids(year, item, sample_size=2000):
    sample_para_id = []
    search_pattern = f"{year}"
    para_id_pattern = f"*_{item}_*"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for file_name in os.listdir(FORMMATED_DIR):
        if file_name.startswith(search_pattern):
            with open(os.path.join(FORMMATED_DIR, file_name), "r") as f:
                for line in f:
                    data = json.loads(line)
                    if fnmatch.fnmatch(data["id"], para_id_pattern):
                        tokens = tokenizer.tokenize(data["contents"])
                        if len(tokens) <= 15 or len(tokens) >= 250:
                            continue
                        sample_para_id.append(data["id"])
    
    print("Number of paragraphs:", len(sample_para_id))
    # Sample 2000 para_ids if the list is larger than the sample size
    if len(sample_para_id) > sample_size:
        sample_para_id = random.sample(sample_para_id, sample_size)

    return sample_para_id

def get_ref_corpus(cik, start_year, end_year):
    ref_id = []
    ref_contents = []
    for year in range(start_year, end_year+1):
        try:
            file_name = get_10K_file_name(cik, year)
            with open(f"{FORMMATED_DIR}/{file_name}", "r") as f:
                for line in f:
                    data = json.loads(line)
                    ref_id.append(data["id"])
                    ref_contents.append(data["contents"])
        except FileNotFoundError:
            continue
    return ref_id, ref_contents

def construct_positive_pairs(sample_para_ids, start_year, end_year, positive_number=5):
    with open("positive_pairs.jsonl", "w") as f:
        # 20230224_10-K_1306830_part1_item1_para5
        for para_id in sample_para_ids:
            target_para = retrieve_paragraph_from_docid(para_id)

            cik = para_id.split("_")[2]
            ref_id, ref_contents = get_ref_corpus(cik, start_year, end_year)

            similarity_scores = tfidf_similarity(ref_contents, target_para)
            top_indices = np.argsort(similarity_scores)[::-1][:positive_number+1]
            top_indices = top_indices[1:] # Skip the top1 index
            
            for index in top_indices:
                # training_pairs.append([target_para, ref_contents[index]])
                pair = {'target': target_para, 'reference': ref_contents[index]}
                f.write(json.dumps(pair) + "\n")

def construct_negative_pairs(sample_para_ids, start_year, end_year, negative_number=5):
    with open("negative_pairs.jsonl", "w") as f:
        # 20230224_10-K_1306830_part1_item1_para5
        for para_id in sample_para_ids:
            target_para = retrieve_paragraph_from_docid(para_id)

            cik = para_id.split("_")[2]
            ref_id, ref_contents = get_ref_corpus(cik, start_year, end_year)

            similarity_scores = tfidf_similarity(ref_contents, target_para)
            top_indices = np.argsort(similarity_scores)[::-1][10:]
            
            negative_indices = random.sample(list(top_indices), negative_number)

            for index in negative_indices:
                # training_pairs.append([target_para, ref_contents[index]])
                pair = {'target': target_para, 'reference': ref_contents[index]}
                f.write(json.dumps(pair) + "\n")

def main():
    sample_para_ids = sample_paragraph_ids(args.sample_year, args.sample_item, args.sample_size)
    
    print(f"Start constructing positive pairs with sample size: {args.sample_size}, positive number: {args.positive_number}")
    construct_positive_pairs(sample_para_ids, args.start_year, args.sample_year, args.positive_number)
    
    print(f"Start constructing negative pairs with sample size: {args.sample_size}, negative number: {args.negative_number}")
    construct_negative_pairs(sample_para_ids, args.start_year, args.sample_year, args.negative_number)

if __name__ == "__main__":
    main()
    