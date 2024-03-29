import os
import json
import fnmatch
import argparse
from functools import partial
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from models import DenseDocumentRetriever, SparseDocumentRetriever, output_hits
from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR
from utils import get_10K_file_name, convert_docid_to_title, retrieve_paragraph_from_docid

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, choices=["dense", "sparse"])
parser.add_argument("--d_encoder", type=str, required=False, help="Document encoder model name") # facebook/dpr-ctx_encoder-multiset-base; sentence-transformers/all-MiniLM-L6-v2
parser.add_argument("--q_encoder", type=str, required=False, help="Query encoder model name") # facebook/dpr-question_encoder-multiset-base; sentence-transformers/all-MiniLM-L6-v2
parser.add_argument("--index_type", type=str, required=True, choices=["multi_fields", "sparse_title", "basic", "meta_data", "title", "ner", "ner_concat"])
parser.add_argument("--cik", type=str, required=True)
parser.add_argument("--target_year", type=str, required=True)
parser.add_argument("--target_item", type=str, required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--target_paragraph", type=str, default=None) # para5
parser.add_argument("--filter_name", type=str, default=None, help="Filter name for the index")
parser.add_argument("--post_filter", action='store_true', default=False, help="Indicates whether to conduct post filtering")
parser.add_argument("--output_jsonl_results", action='store_true', default=False) # ouput the retrieval results to view the retrieved paragraphs

args = parser.parse_args()

model_mapping = {
    "dense": DenseDocumentRetriever,
    "sparse": SparseDocumentRetriever
}

sparse_index_mapping = {
    "multi_fields": "multi_fields",
    "sparse_title": "sparse_title", 
    "ner": "ner"
}

dense_index_mapping = {
    "basic": "basic",
    "meta_data": "meta_data",
    "title": "title", 
    "ner_concat": "ner_concat"
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
        
        encoder_name = args.d_encoder.split('/')[-1] # potential bug
        index_name = f"{encoder_name}-{dense_index_mapping[index_type]}"

    if filter_name is not None:
        index_name += f"-{filter_name}"

    return index_name
    

def get_retriever(model_type, index_type, k, filter_name=None):
    if model_type not in model_mapping:
        raise ValueError(f"Unknown model type: {model_type}")
    
    index_name = get_index_name(model_type, index_type, filter_name)
    print(f"Using index: {index_name}")

    if model_type == "sparse":
        searcher = LuceneSearcher(f"{INDEX_DIR}/{index_name}")
        # TODO: check if NER is supported
        if index_type == "ner":
            return SparseDocumentRetriever(searcher, k=k, fields={"ORG": 0.4, "contents": 0.6})
        return SparseDocumentRetriever(searcher, k=k)

    elif model_type == "dense":
        # query_encoder = DprQueryEncoder("facebook/dpr-question_encoder-multiset-base")
        # searcher = FaissSearcher(f"{INDEX_DIR}/{index_name}", query_encoder)
        print(f"Query encoder: {args.q_encoder}")
        searcher = FaissSearcher(f"{INDEX_DIR}/{index_name}", args.q_encoder)
        return DenseDocumentRetriever(searcher, k=k)


def output_hits_trec(hits, target_id, output_file, index_name):
    # {query_id} Q0 {doc_id} {rank} {score} {index_name}
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # if filter_name is not None:
    #     index_name = f"{index_type}-{filter_name}"
    # else:
    #     index_name = index_type

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
    model_type = args.model
    index_type = args.index_type
    filter_name = args.filter_name
    print(model_type, index_type, filter_name)

    retriever = get_retriever(model_type, index_type, k=args.k, filter_name=filter_name)

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
    
    if args.post_filter:
        partial_filter_function = partial(filter_function, cik=cik, item=target_item, start_year=target_year, end_year=target_year, filter_out=True)
    
    base_target_file_dir = os.path.dirname(FORMMATED_DIR)
    preprocessed_target_file_dir = os.path.join(base_target_file_dir, index_type)
    print("preprocessed_target_file_dir:", preprocessed_target_file_dir)
    # with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
    with open(os.path.join(preprocessed_target_file_dir, target_file_name), "r") as open_file: # let the target paragraph be the same format as the index
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")
                
                target_title = convert_docid_to_title(data["id"]) # retrieve時沒有concat title
                target_paragraph_content = data["contents"]
                
                full_query = f"{instruction}; {target_paragraph_content}"
                # print(full_query)
                if args.post_filter:
                    hits = retriever.search_documents(full_query, filter_function=partial_filter_function)
                else:
                    hits = retriever.search_documents(full_query)

                index_name = get_index_name(model_type, index_type, filter_name)
                # index_type_with_filter = f"{index_type}-{filter_name}" if filter_name is not None else index_type

                output_file_trec = os.path.join('retrieval_results_trec', f"{cik}_{target_year}", index_name, data["id"] + '.txt')
                output_hits_trec(hits, data["id"], output_file_trec, index_name)

                if args.output_jsonl_results:
                    output_file = os.path.join('retrieval_results', f"{cik}_{target_year}", index_name, data["id"] + '.jsonl')
                    output_hits(hits, output_file)

if __name__ == "__main__":
    main()
    # for commit purpose