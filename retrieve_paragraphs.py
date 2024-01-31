import os
import json
import fnmatch
import argparse
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from models import DenseDocumentRetriever, SparseDocumentRetriever, output_hits
from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR
from utils import get_10K_file_name, convert_docid_to_title, retrieve_paragraph_from_docid

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, choices=["dense", "sparse"])
parser.add_argument("--index_type", type=str, required=True)
parser.add_argument("--cik", type=str, required=True)
parser.add_argument("--target_year", type=str, required=True)
parser.add_argument("--target_item", type=str, required=True)
parser.add_argument("--target_paragraph", type=str, default=None) # para5
parser.add_argument("--filter_name", type=str, default=None)

args = parser.parse_args()

model_mapping = {
    "dense": DenseDocumentRetriever,
    "sparse": SparseDocumentRetriever
}

sparse_index_mapping = {
    "multi_fields": "multi_fields",
    "sparse_title": "sparse_title", 
}

dense_index_mapping = {
    "basic": "dpr-ctx_encoder-multiset-base-basic",
    "meta_data": "dpr-ctx_encoder-multiset-base-meta_data",
    "title": "dpr-ctx_encoder-multiset-base-title"
}

with open(os.path.join(ROOT, 'collections', 'cik_to_company.json'), 'r') as f:
    cik_to_company = json.load(f)

with open(os.path.join(ROOT, 'collections', 'item_mapping.json'), 'r') as f:
    item_mapping = json.load(f)

def convert_docid_to_title(docid):
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


def get_index_name(model_type, index_type, filter_name=None):
    if model_type not in model_mapping:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_type == "sparse":
        if index_type not in sparse_index_mapping:
            raise ValueError(f"Invalid index: {index_type} for sparse retriever")
        index_name = sparse_index_mapping[index_type]
    
    elif model_type == "dense":
        if index_type not in dense_index_mapping:
            raise ValueError(f"Invalid index: {index_type} for dense retriever")
        index_name = dense_index_mapping[index_type]

    if filter_name is not None:
        index_name += f"-{filter_name}"

    return index_name
    

def get_retriever(model_type, index_type, filter_name=None):
    if model_type not in model_mapping:
        raise ValueError(f"Unknown model type: {model_type}")
    
    index_name = get_index_name(model_type, index_type, filter_name)
    print(f"Using index: {index_name}")

    if model_type == "sparse":
        searcher = LuceneSearcher(f"{INDEX_DIR}/{index_name}")
        return SparseDocumentRetriever(searcher)

    elif model_type == "dense":
        query_encoder = DprQueryEncoder("facebook/dpr-question_encoder-multiset-base")
        searcher = FaissSearcher(f"{INDEX_DIR}/{index_name}", query_encoder)
        return DenseDocumentRetriever(searcher, query_encoder.tokenizer)

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
    model_type = args.model
    index_type = args.index_type
    filter_name = args.filter_name
    print(model_type, index_type, filter_name)

    retriever = get_retriever(model_type, index_type, filter_name=filter_name)

    cik = args.cik
    target_company = cik_to_company[cik]
    target_item = args.target_item
    target_paragraph = args.target_paragraph
    target_year = args.target_year

    # instruction should be different for different model type
    if model_type == "dense":
        instruction = f"Find paragraphs that are relevant to this paragraph from {target_company}"
    elif model_type == "sparse":
        instruction = f"company_name: {target_company}"

    target_file_name = get_10K_file_name(cik, target_year)

    if target_paragraph:
        search_pattern = f"*_{target_item}_{target_paragraph}"
    else:
        search_pattern = f"*_{target_item}_para*" # "20220426_10-Q_789019_part1_item2_para492"
    
    with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")
                
                target_title = convert_docid_to_title(data["id"])
                target_paragraph = data["contents"]
                
                full_query = f"{instruction}; {target_paragraph}"
                hits = retriever.search_documents(full_query)

                index_type_with_filter = f"{index_type}-{filter_name}" if filter_name is not None else index_type

                output_file = os.path.join(ROOT, 'retrieval_results', f"{cik}_{target_year}", index_type_with_filter, data["id"] + '.jsonl')
                output_hits(hits, output_file)

if __name__ == "__main__":
    main()