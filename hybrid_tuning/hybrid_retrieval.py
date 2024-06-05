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
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher
from sentence_transformers import SentenceTransformer

from config import ROOT, FORMMATED_DIR, INDEX_DIR
from config import CIK_TO_COMPANY, CIK_TO_SECTOR, ITEM_MAPPING
from utils import get_10K_file_name, retrieve_paragraph_from_docid, convert_docid_to_title
from models import HybridDocumentRetriever, CustomSentenceTransformerEncoder

print(f"FORMMATED_DIR: {FORMMATED_DIR}")
print(f"INDEX_DIR: {INDEX_DIR}")
print(f"Current working directory: {os.getcwd()}")

sparse_index_mapping = {
    "multi_fields": "multi_fields",
    "ner": "ner", 
    "company_name": "company_name", 
    "title": "title", 
    "basic": "basic"
}

fields_mapping = {
    "multi_fields": {"company_name": 0.4, "sector": 0.2, "contents": 0.4},
    "ner": {"ORG": 0.4, "contents": 0.6}
}

dense_index_mapping = {
    "basic": "basic",
    "title": "title", 
    "company_name": "company_name", 
    "ner_concat": "ner_concat", 
    "ner_concat_with_company_name": "ner_concat_with_company_name", 
    "ner_concat_with_title": "ner_concat_with_title"
}

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

def get_sparse_index_name(index_type, filter_name=None):
    if index_type not in sparse_index_mapping:
        raise ValueError(f"Invalid index: {index_type} for sparse retriever")
    index_name = sparse_index_mapping[index_type]

    if filter_name:
        index_name = f"{index_name}-{filter_name}"
    
    return index_name

def get_dense_index_name(index_type, d_encoder, filter_name=None):
    if index_type not in dense_index_mapping:
        raise ValueError(f"Invalid index: {index_type} for dense retriever")
    
    encoder_name = d_encoder.split("/")[-1]
    index_name = f"{encoder_name}-{dense_index_mapping[index_type]}"

    if filter_name:
        index_name = f"{index_name}-{filter_name}"
    
    return index_name

def get_sparse_searcher(index_type, k1, b, filter_name=None):
    index_name = get_sparse_index_name(index_type, filter_name)
    index_path = os.path.join(INDEX_DIR, index_name)
    print(f"Using index: {index_name}")
    print(f"Index path: {index_path}")

    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=k1, b=b)

    return searcher

def get_dense_searcher(index_type, d_encoder, q_encoder, filter_name=None):
    index_name = get_dense_index_name(index_type, d_encoder, filter_name)
    index_path = os.path.join(INDEX_DIR, index_name)
    print(f"Using index: {index_name}")
    print(f"Index path: {index_path}")

    # create seracher
    searcher = None
    if os.path.isdir(q_encoder):
        print(f"load the fine-tuned SentenceTransformer model from {q_encoder}")
        query_encoder = CustomSentenceTransformerEncoder(q_encoder)
        searcher = FaissSearcher(index_path, query_encoder)
    else:
        print(f"load the pre-trained query encoder from huggingface model hub: {q_encoder}")
        searcher = FaissSearcher(index_path, q_encoder)
    
    return searcher

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepend_info", choices=["instruction", "target_title", "target_company", "null"], default="null", help='Specify the type of information to prepend to the target paragraph')
    parser.add_argument("--cik", type=str, required=True)
    parser.add_argument("--target_year", type=str, required=True)
    parser.add_argument("--target_item", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target_paragraph", type=str, default=None)
    parser.add_argument("--filter_name", type=str, default=None, help="Filter name for the index")
    parser.add_argument("--post_filter", action='store_true', default=False, help="Indicates whether to conduct post filtering")
    # for sparse retriever
    parser.add_argument("--sparse_index_type", type=str, required=True, choices=["multi_fields", "basic", "ner", "company_name", "title"])
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b parameter")
    # for dense retriever
    parser.add_argument("--dense_index_type", type=str, required=True, choices=["basic", "meta_data", "title", "ner_concat", "company_name"])
    parser.add_argument("--d_encoder", type=str, required=False, help="Document encoder model name") # facebook/dpr-ctx_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2
    parser.add_argument("--q_encoder", type=str, required=False, help="Query encoder model name") # facebook/dpr-question_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2
    # for hybrid
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for the hybrid retriever")
    parser.add_argument("--weight_on_dense", action='store_true', default=False, help="Indicates whether to use the weight on dense retriever for the hybrid retriever")

    args = parser.parse_args()

    s_searcher = get_sparse_searcher(args.sparse_index_type, args.k1, args.b, args.filter_name)
    d_searcher = get_dense_searcher(args.dense_index_type, args.d_encoder, args.q_encoder, args.filter_name)
    h_searcher = HybridSearcher(s_searcher, d_searcher)

    retriever = HybridDocumentRetriever(h_searcher, k=args.k, alpha=args.alpha, weight_on_dense=args.weight_on_dense)

    target_file_name = get_10K_file_name(args.cik, args.target_year)

    # {yyyymmdd}_{form}_{CIK}_part{part number}_item{item number}_para{paragraph number}
    if args.target_paragraph:
        search_pattern = f"*_{args.target_item}_{args.target_paragraph}"
    else:
        search_pattern = f"*_{args.target_item}_para*"

    if args.post_filter:
        partial_filter_function = partial(filter_function, cik=args.cik, item=args.target_item, start_year=args.target_year, end_year=args.target_year, filter_out=True)  # filter out the cik's target_item in target_year

    target_company = CIK_TO_COMPANY[args.cik]
    with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")

                target_paragraph_content = data["contents"]

                if args.prepend_info == "null":
                    full_query = target_paragraph_content
                elif args.prepend_info == "instruction":
                    instruction = f"Find paragraphs that are relevant to this paragraph from {target_company}"
                    full_query = f"{instruction}; {target_paragraph_content}"
                elif args.prepend_info == "target_company":
                    full_query = f"{target_company}; {target_paragraph_content}"
                elif args.prepend_info == "target_title":
                    target_title = convert_docid_to_title(data["id"])
                    full_query = f"{target_title}; {target_paragraph_content}"
                else:
                    raise ValueError(f"Invalid prepend_info: {args.prepend_info}")
                
                if args.post_filter:
                    hits = retriever.search_documents(full_query, filter_function=partial_filter_function)
                else:
                    hits = retriever.search_documents(full_query)

                s_index_name = get_sparse_index_name(args.sparse_index_type, args.filter_name)
                d_index_name = get_dense_index_name(args.dense_index_type, args.d_encoder, args.filter_name)

                if args.weight_on_dense:
                    retrieval_system_tag = f"hybrid-{args.sparse_index_type}-{d_index_name}-{args.prepend_info}-alpha_{args.alpha}-weight_on_dense"
                else:
                    retrieval_system_tag = f"hybrid-{args.sparse_index_type}-{d_index_name}-{args.prepend_info}-alpha_{args.alpha}"

                output_file_trec = os.path.join("retrieval_results_trec", f"{args.cik}_{args.target_year}", retrieval_system_tag, data["id"] + '.txt')
                output_hits_trec(hits, data["id"], output_file_trec, retrieval_system_tag)

                output_file = os.path.join('retrieval_results', f"{args.cik}_{args.target_year}", retrieval_system_tag, data["id"] + '.jsonl')
                output_hits(hits, output_file)

if __name__ == "__main__":
    main()