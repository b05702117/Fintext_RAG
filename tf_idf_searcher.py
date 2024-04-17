import os
import json
import fnmatch
import argparse
import numpy as np
from functools import partial

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR
from utils import get_10K_file_name, convert_docid_to_title, retrieve_paragraph_from_docid

class TfIdfRetriever:
    def __init__(self, ref_ids, ref_contents, k=10):
        self.ref_ids = ref_ids
        self.ref_contents = ref_contents
        self.k = k

    @staticmethod
    def tfidf_similarity(documents: list[str], query: str) -> np.ndarray:
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


    def search_documents(self, query: str, filter_function=None) -> list[dict]:
        similarity_scores = self.tfidf_similarity(self.ref_contents, query)
        sorted_indices = np.argsort(similarity_scores)[::-1]

        filtered_hits = []
        
        for idx in sorted_indices[1:]: # skip the first idx since it is identical to the query
            hit = {
                "docid": self.ref_ids[idx],
                "score": similarity_scores[idx]
            }
            
            if filter_function:
                if filter_function(hit["docid"]):
                    filtered_hits.append(hit)
                else:
                    continue
            else:
                filtered_hits.append(hit)

            if len(filtered_hits) >= self.k:
                break
        
        return filtered_hits


def filter_function(docid, cik, item, start_year, end_year, filter_out=True):
    docid_componenets = docid.split('_')
    doc_year = int(docid_componenets[0][:4])
    doc_cik = docid_componenets[2]
    doc_item = docid_componenets[4]

    # Check criteria
    match_cik = str(cik) == doc_cik if cik else True
    match_year = (int(start_year) <= doc_year <= (int(end_year))) if start_year and end_year else True
    match_item = str(item) == doc_item if item else True

    match_criteria = match_cik and match_year and match_item

    if match_criteria and not filter_out:   # write the hits that match the criteria if `filter_out` is not set (standard filtering)
        return True
    elif not match_criteria and filter_out: # write the hits that do not match the criteria if `filter_out` is set (exclude the lines that match the criteria)
        return True
    return False


def output_hits_trec(hits, target_id, output_file, index_name):
    # {query_id} Q0 {doc_id} {rank} {score} {index_name}
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for i in range(len(hits)):
            f.write(f"{target_id} Q0 {hits[i]['docid']} {i+1} {hits[i]['score']} {index_name}\n")


def output_hits(hits, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for hit in hits:
            result = {
                'id': hit['docid'],
                'score': float(hit['score']), # convert float32 to standard float
                'contents': retrieve_paragraph_from_docid(hit['docid'])
            }
            json.dump(result, f)
            f.write('\n')

def generate_corpus_self(cik: str, start_year: int, end_year: int, corpus_dir: str=FORMMATED_DIR) -> tuple[list[str], list[str]]:
    ref_ids, ref_contents = [], []

    for year in range(start_year, end_year + 1):
        ref_file_name = get_10K_file_name(cik, year)
        with open(f"{corpus_dir}/{ref_file_name}", "r") as f:
            for line in f:
                data = json.loads(line)
                ref_ids.append(data["id"])
                ref_contents.append(data["contents"])

    return ref_ids, ref_contents

def generate_corpus(start_year: int, end_year: int, corpus_dir: str=FORMMATED_DIR) -> tuple[list[str], list[str]]:
    ref_ids, ref_contents = [], []

    for year in range(start_year, end_year + 1):
        search_pattern = f"{year}"
        for file_name in os.listdir(corpus_dir):
            if file_name.startswith(search_pattern):
                with open(f"{corpus_dir}/{file_name}", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        ref_ids.append(data["id"])
                        ref_contents.append(data["contents"])

    return ref_ids, ref_contents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", type=str, required=True)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--target_year", type=int, required=True)
    parser.add_argument("--target_item", type=str, required=True)
    parser.add_argument("--target_paragraph", type=str, default=None) # para5
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--self_reference", action='store_true', default=False, help="Indicates whether to conduct post filtering")
    parser.add_argument("--post_filter", action='store_true', default=False, help="Indicates whether to conduct post filtering")
    parser.add_argument("--output_jsonl_results", action='store_true', default=False) # ouput the retrieval results to view the retrieved paragraphs

    args = parser.parse_args()

    target_file_name = get_10K_file_name(args.cik, args.target_year)

    if args.target_paragraph:
        search_pattern = f"*_{args.target_item}_{args.target_paragraph}"
    else:
        search_pattern = f"*_{args.target_item}_para*"

    if args.post_filter:
        partial_filter_function = partial(filter_function, cik=args.cik, item=args.target_item, start_year=args.target_year, end_year=args.target_year, filter_out=True) # filter out the cik's target_item in target_year
    
    if args.self_reference:
        ref_ids, ref_contents = generate_corpus_self(args.cik, args.start_year, args.target_year)
        retriever = TfIdfRetriever(ref_ids, ref_contents, k=args.k)
    else:
        ref_ids, ref_contents = generate_corpus(args.start_year, args.target_year)
        retriever = TfIdfRetriever(ref_ids, ref_contents, k=args.k)

    index_name = "tfidf" # the name of the retrieval system
    if args.self_reference:
        index_name += "-self_reference"

    with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")
                
                target_paragraph_content = data["contents"]

                if args.post_filter:
                    hits = retriever.search_documents(target_paragraph_content, filter_function=partial_filter_function)
                else:
                    hits = retriever.search_documents(target_paragraph_content)
                
                output_file_trec = os.path.join('retrieval_results_trec', f"{args.cik}_{args.target_year}", index_name, data["id"] + '.txt')
                output_hits_trec(hits, data["id"], output_file_trec, index_name)

                if args.output_jsonl_results:
                    output_file = os.path.join('retrieval_results', f"{args.cik}_{args.target_year}", index_name, data["id"] + '.jsonl')
                    output_hits(hits, output_file)

if __name__ == "__main__":
    main()