import os
import json
import fnmatch
import argparse
from functools import partial
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from models import CustomHybridSearcher # CustomHybridSearcher inherits from HybridSearcher and overrides the search method to include the fields parameter.
from models import DenseDocumentRetriever, SparseDocumentRetriever, HybridDocumentRetriever, output_hits
from models import CustomSentenceTransformerEncoder
from config import FORMMATED_DIR, INDEX_DIR, CIK_TO_COMPANY
from utils import get_10K_file_name, convert_docid_to_title, retrieve_paragraph_from_docid

parser = argparse.ArgumentParser()

parser.add_argument("--retriever_type", type=str, required=True, choices=["dense", "sparse", "hybrid"])
parser.add_argument("--index_type", type=str, required=True, choices=["multi_fields", "sparse_title", "basic", "meta_data", "title", "ner", "ner_concat"]) # TODO: naming 改成index_format
parser.add_argument("--prepend_info", type=str, choices=["instruction", "target_title", "target_company", "null"], default="target_title", help="Information to prepend to the target paragraph")
parser.add_argument("--cik", type=str, required=True)
parser.add_argument("--target_year", type=str, required=True)
parser.add_argument("--target_item", type=str, required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--target_paragraph", type=str, default=None) # para5
parser.add_argument("--filter_name", type=str, default=None, help="Filter name for the index")
parser.add_argument("--post_filter", action='store_true', default=False, help="Indicates whether to conduct post filtering")
parser.add_argument("--output_jsonl_results", action='store_true', default=False) # ouput the retrieval results to view the retrieved paragraphs
# for sparse retriever
parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
parser.add_argument("--b", type=float, default=0.9, help="BM25 b parameter")
# for dense retriever
parser.add_argument("--d_encoder", type=str, required=False, help="Document encoder checkpoint name") # facebook/dpr-ctx_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2
parser.add_argument("--q_encoder", type=str, required=False, help="Query encoder checkpoint name") # facebook/dpr-question_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2
# for hybrid retriever
parser.add_argument("--sparse_index_type", type=str, choices=["multi_fields", "basic", "ner", "company_name", "title"], default="multi_fields", help="Index type for the sparse retriever")
parser.add_argument("--sparse_filter_name", type=str, default=None, help="Filter name for the sparse index in the hybrid retriever")
parser.add_argument("--weight_on_dense", action='store_true', default=False, help="Indicates whether to use the weight on dense retriever for the hybrid retriever")
parser.add_argument("--alpha", type=float, default=0.1, help="Weight for the hybrid retriever")

args = parser.parse_args()

# not used
retriever_mapping = {
    "dense": DenseDocumentRetriever,
    "sparse": SparseDocumentRetriever, 
    "hybrid": HybridDocumentRetriever
}

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

dense_index_mapping = {
    "basic": "basic",
    "meta_data": "meta_data",
    "title": "title", 
    "ner_concat": "ner_concat", 
    "company_name": "company_name"
}


def get_index_name(retriever_type, index_type, filter_name=None):
    # TODO: training要改檔名
    if retriever_type not in retriever_mapping:
        raise ValueError(f"Unknown retriever_type type: {retriever_type}")
    
    if retriever_type == "sparse":
        if index_type not in sparse_index_mapping:
            raise ValueError(f"Invalid index: {index_type} for sparse retriever")
        index_name = sparse_index_mapping[index_type]
    
    elif retriever_type == "dense":
        if index_type not in dense_index_mapping:
            raise ValueError(f"Invalid index: {index_type} for dense retriever")
        
        encoder_name = args.d_encoder.split('/')[-1] # potential bug
        index_name = f"{encoder_name}-{dense_index_mapping[index_type]}"
    
    elif retriever_type == "hybrid":
        if index_type not in dense_index_mapping:
            raise ValueError(f"Invalid index: {index_type} for dense retriever")
        
        encoder_name = args.d_encoder.split('/')[-1] # potential bug
        index_name = f"{encoder_name}-{dense_index_mapping[index_type]}"

    if filter_name is not None:
        index_name += f"-{filter_name}"

    return index_name
    
def create_searcher(retriever_type, index_path, q_encoder=None):
    if retriever_type == "sparse":
        searcher = LuceneSearcher(index_path)
        searcher.set_bm25(k1=float(args.k1), b=float(args.b))
        return searcher
    elif retriever_type == "dense":
        if os.path.isdir(q_encoder):
            print(f"load the fine-tuned SentenceTransformer model from {q_encoder}")
            query_encoder = CustomSentenceTransformerEncoder(q_encoder)
            return FaissSearcher(index_path, query_encoder)
        else:
            print(f"load the pre-trained query encoder from huggingface model hub: {q_encoder}")
            return FaissSearcher(index_path, q_encoder)
    else:
        raise ValueError(f"Unknown model type: {retriever_type}")

def get_retriever(retriever_type, index_type, k, filter_name=None):
    if retriever_type not in retriever_mapping:
        raise ValueError(f"Unknown model type: {retriever_type}")
    
    index_name = get_index_name(retriever_type, index_type, filter_name)
    index_path = os.path.join(INDEX_DIR, index_name)
    print(f"Using index: {index_name}")
    print(f"Index path: {index_path}")

    if retriever_type == "sparse":
        searcher = create_searcher("sparse", index_path)
        fields = fields_mapping.get(index_type, None)
        return SparseDocumentRetriever(searcher, k=k, fields=fields)
    
    elif retriever_type == "dense":
        searcher = create_searcher("dense", index_path, args.q_encoder)
        return DenseDocumentRetriever(searcher, k=k)

    elif retriever_type == "hybrid":
        # suppose the index type is for dense retriever
        sparse_index_name = get_index_name("sparse", args.sparse_index_type, args.sparse_filter_name if args.sparse_filter_name else filter_name)
        s_searcher = create_searcher("sparse", os.path.join(INDEX_DIR, sparse_index_name))
        d_searcher = create_searcher("dense", index_path, args.q_encoder)
        h_searcher = CustomHybridSearcher(d_searcher, s_searcher) # TODO: modify the HybridSearcher to accept the fields for sparse searcher
        return HybridDocumentRetriever(h_searcher, k=k, alpha=args.alpha, weight_on_dense=args.weight_on_dense, fields=fields_mapping.get(args.sparse_index_type, None))

def output_hits_trec(hits, target_id, output_file, index_name):
    # {query_id} Q0 {doc_id} {rank} {score} {index_name}
    os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure the directory exists

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

def generate_query(docid, content, company_name, prepend_info):
    if prepend_info == "null":
        return content
    elif prepend_info == "instruction":
        instruction = f"Find paragraphs that are relevant to this paragraph from {company_name}"
        return f"{instruction}; {content}"
    elif prepend_info == "target_title":
        target_title = convert_docid_to_title(docid)
        return f"{target_title}; {content}"
    elif prepend_info == "target_company":
        return f"{company_name}; {content}"
    else:
        raise ValueError(f"Invalid prepend_info: {args.prepend_info}")

def main():
    print(args.retriever_type, args.index_type, args.filter_name)

    retriever = get_retriever(args.retriever_type, args.index_type, k=args.k, filter_name=args.filter_name)

    target_company = CIK_TO_COMPANY[args.cik]
    target_file_name = get_10K_file_name(args.cik, args.target_year)

    if args.target_paragraph:
        search_pattern = f"*_{args.target_item}_{args.target_paragraph}"
    else:
        search_pattern = f"*_{args.target_item}_para*" # "20220426_10-Q_789019_part1_item2_para492"
    
    if args.post_filter:
        partial_filter_function = partial(filter_function, cik=args.cik, item=args.target_item, start_year=args.target_year, end_year=args.target_year, filter_out=True) # filter out the cik's target_item in target_year
    
    base_target_file_dir = os.path.dirname(FORMMATED_DIR)
    preprocessed_target_file_dir = os.path.join(base_target_file_dir, args.index_type)
    print("preprocessed_target_file_dir:", preprocessed_target_file_dir)
    with open(os.path.join(FORMMATED_DIR, target_file_name), "r") as open_file:
    # with open(os.path.join(preprocessed_target_file_dir, target_file_name), "r") as open_file: # let the target paragraph be the same format as the index
        for line in open_file:
            data = json.loads(line)
            if fnmatch.fnmatch(data["id"], search_pattern):
                print(f"start searching for {data['id']}")
                
                full_query = generate_query(data["id"], data["contents"], target_company, args.prepend_info)
                # print(full_query)

                if args.post_filter:
                    hits = retriever.search_documents(full_query, filter_function=partial_filter_function)
                else:
                    hits = retriever.search_documents(full_query)

                index_name = get_index_name(args.retriever_type, args.index_type, args.filter_name)
                if args.retriever_type == "hybrid":
                    s_index_name = get_index_name("sparse", args.sparse_index_type, args.sparse_filter_name if args.sparse_filter_name else args.filter_name)
                    retrieval_system_tag = f"hybrid-{s_index_name}-{index_name}"
                    if args.weight_on_dense:
                        retrieval_system_tag += "-weight_on_dense"
                else:
                    retrieval_system_tag = f"{index_name}"

                output_file_trec = os.path.join('retrieval_results_trec', f"{args.cik}_{args.target_year}", retrieval_system_tag, data["id"] + '.txt')
                output_hits_trec(hits, data["id"], output_file_trec, retrieval_system_tag)

                if args.output_jsonl_results:
                    output_file = os.path.join('retrieval_results', f"{args.cik}_{args.target_year}", retrieval_system_tag, data["id"] + '.jsonl')
                    output_hits(hits, output_file)

if __name__ == "__main__":
    main()