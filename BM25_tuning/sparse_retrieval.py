import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import os
import json
import fnmatch
import argparse
from functools import partial

from pyserini.search.lucene import LuceneSearcher

from config import ROOT, FORMMATED_DIR, INDEX_DIR
from models import SparseDocumentRetriever
from utils import get_10K_file_name, retrieve_paragraph_from_docid, convert_docid_to_title
# from retrieve_paragraphs import filter_function, output_hits, output_hits_trec

print(f"FORMMATED_DIR: {FORMMATED_DIR}")
print(f"INDEX_DIR: {INDEX_DIR}")
print(f"Current working directory: {os.getcwd()}")

sparse_index_mapping = {
    "multi_fields": "multi_fields",
    "sparse_title": "sparse_title", 
    "ner": "ner", 
    "company_name": "company_name", 
    "title": "title", 
    "basic": "basic"
}

fields_mapping = {
    "multi_fields": {"company_name": 0.4, "sector": 0.2, "contents": 0.4},
    "ner": {"ORG": 0.4, "contents": 0.6}
}

with open(os.path.join(ROOT, 'collections', 'cik_to_company.json'), 'r') as f:
    cik_to_company = json.load(f)

with open(os.path.join(ROOT, 'collections', 'item_mapping.json'), 'r') as f:
    item_mapping = json.load(f)


def get_index_name(index_type, filter_name=None):
    if index_type not in sparse_index_mapping:
        raise ValueError(f"Invalid index: {index_type} for sparse retriever")
    index_name = sparse_index_mapping[index_type]

    if filter_name:
        index_name = f"{index_name}-{filter_name}"
    
    return index_name

def get_retriever(index_type, k1, b, k=10, filter_name=None):
    index_name = get_index_name(index_type, filter_name)
    index_path = os.path.join(INDEX_DIR, index_name)
    print(f"Using index: {index_name}")
    print(f"Index path: {index_path}")
    
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=k1, b=b)

    fields = fields_mapping.get(index_type, None) 
    if fields:
        print(f"Using fields for: {index_type}")

    return SparseDocumentRetriever(searcher, k=k, fields=fields)

def output_hits_trec(hits, target_id, output_file, index_name):
    # {query_id} Q0 {doc_id} {rank} {score} {index_name}
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for i in range(len(hits)):
            f.write(f"{target_id} Q0 {hits[i].docid} {i+1} {hits[i].score} {index_name}\n")

def output_hits(hits, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for hit in hits:
            result = {
                'id': hit.docid,
                'score': float(hit.score), # convert float32 to standard float
                'contents': retrieve_paragraph_from_docid(hit.docid)
            }
            json.dump(result, f)
            f.write('\n')


def filter_function(hits, cik, item, start_year, end_year, filter_out=True):
    filtered_hits = []
    for hit in hits:
        docid = hit.docid
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
            filtered_hits.append(hit)
        elif not match_criteria and filter_out: # write the hits that do not match the criteria if `filter_out` is set (exclude the lines that match the criteria)
            filtered_hits.append(hit)
    return filtered_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_type", type=str, required=True, choices=["multi_fields", "basic", "ner", "company_name", "title"])
    parser.add_argument("--prepend_info", choices=["instruction", "target_title", "target_company", "null"], default="null", help='Specify the type of information to prepend to the target paragraph')
    parser.add_argument("--cik", type=str, required=True)
    parser.add_argument("--target_year", type=str, required=True)
    parser.add_argument("--target_item", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target_paragraph", type=str, default=None)
    parser.add_argument("--filter_name", type=str, default=None, help="Filter name for the index")
    parser.add_argument("--post_filter", action='store_true', default=False, help="Indicates whether to conduct post filtering")
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b parameter")

    args = parser.parse_args()

    retriever = get_retriever(args.index_type, args.k1, args.b, args.k, args.filter_name)

    target_file_name = get_10K_file_name(args.cik, args.target_year)

    if args.target_paragraph:
        search_pattern = f"*_{args.target_item}_{args.target_paragraph}"
    else:
        search_pattern = f"*_{args.target_item}_para*" # "20220426_10-Q_789019_part1_item2_para492"
    
    if args.post_filter:
        partial_filter_function = partial(filter_function, cik=args.cik, item=args.target_item, start_year=args.target_year, end_year=args.target_year, filter_out=True) # filter out the cik's target_item in target_year


    target_company = cik_to_company[args.cik]
    with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")

                target_paragraph_content = data["contents"]

                if args.prepend_info == "null":
                    full_query = target_paragraph_content
                elif args.prepend_info == "instruction":
                    full_query = f"company name: {target_company}; {target_paragraph_content}"
                elif args.prepend_info == "target_title":
                    target_title = convert_docid_to_title(data["id"])
                    full_query = f"{target_title}; {target_paragraph_content}"
                elif args.prepend_info == "target_company":
                    full_query = f"{target_company}; {target_paragraph_content}"
                else:
                    raise ValueError(f"Invalid prepend_info: {args.prepend_info}")

                if args.post_filter:
                    hits = retriever.search_documents(full_query, filter_function=partial_filter_function)
                else:
                    hits = retriever.search_documents(full_query)
                
                index_name = get_index_name(args.index_type, args.filter_name)

                output_file_trec = os.path.join("retrieval_results_trec", f"{args.cik}_{args.target_year}", f"{index_name}-{args.prepend_info}-k1_{args.k1}-b_{args.b}", data["id"] + '.txt')
                output_hits_trec(hits, data["id"], output_file_trec, f"{index_name}-{args.prepend_info}-k1_{args.k1}-b_{args.b}")

                output_file = os.path.join('retrieval_results', f"{args.cik}_{args.target_year}", f"{index_name}-{args.prepend_info}-k1_{args.k1}-b_{args.b}", data["id"] + '.jsonl')
                output_hits(hits, output_file)

if __name__ == "__main__":
    main()