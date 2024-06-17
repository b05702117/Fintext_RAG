import json
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--retriever_type", type=str, required=True, choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--index_type", type=str, required=True, choices=["multi_fields", "sparse_title", "basic", "meta_data", "title", "ner", "ner_concat"])
    parser.add_argument("--prepend_info", type=str, choices=["instruction", "target_title", "target_company", "null"], default="target_title", help="Information to prepend to the target paragraph")
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

    return args

def run_retriever(args, cik, target_year, target_item, target_paragraph):
    filter_name = f"year2018_{target_year}"
    command = [
        "python", "retrieve_paragraphs.py", 
        "--retriever_type", f"{args.retriever_type}", 
        "--index_type", f"{args.index_type}", 
        "--prepend_info", "target_title", 
        "--cik", f"{cik}", 
        "--target_year", f"{target_year}", 
        "--target_item", f"{target_item}", 
        "--target_paragraph", f"{target_paragraph}", 
        "--filter_name", f"{filter_name}", 
        "--post_filter", 
    ]
    if args.retriever_type == "sparse":
        command.extend(
            [
                "--k1", f"{args.k1}", 
                "--b", f"{args.b}"
            ]
        )
    elif args.retriever_type == "dense":
        command.extend(
            [
                "--d_encoder", f"{args.d_encoder}", 
                "--q_encoder", f"{args.q_encoder}"
            ]
        )
    elif args.retriever_type == "hybrid":
        command.extend(
            [
                "--d_encoder", f"{args.d_encoder}", 
                "--q_encoder", f"{args.q_encoder}"
            ]
        )

    subprocess.run(command, stdout=subprocess.PIPE, text=True)

if __name__ == '__main__':
    file_path = '/home/ybtu/fin.rag/annotation/annotated_result/all/aggregate_qlabels.jsonl'
    args = parse_args()

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            para_id = data['id']

            id_components = para_id.split('_')
            year = id_components[0][:4]
            form = id_components[1]
            cik = id_components[2]
            item = id_components[4]
            para = id_components[5]

            print(f"start retrieving reference for para_id: {para_id}")
            run_retriever(args, cik, year, item, para)