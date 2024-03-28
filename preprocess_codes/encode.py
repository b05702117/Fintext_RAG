# custom encoding functions

import os
import json
import argparse
import torch
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="/tmp2/ybtu/Fintext_RAG/training/model")
parser.add_argument("--data_root_path", type=str, default="/home/ybtu/FinNLP/collections/10K")
parser.add_argument("--data_format_type", type=str, required=True, choices=["multi_fields", "sparse_title", "basic", "meta_data", "title", "ner", "ner_concat"])
parser.add_argument("--output_root_path", type=str, default="../embeddings/10K")
parser.add_argument("--output_file_name", type=str, default="embeddings.jsonl")
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()

def load_documents(data_path):
    documents = []
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".jsonl"):
            continue
        
        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                documents.append({
                    "id": data["id"], 
                    "contents": data["contents"]
                })
    
    return documents

def encode_documents(model, documents, output_file, batch_size=16):
    with open(output_file, "w") as f:
        for start_idx in range(0, len(documents), batch_size):
            batch = documents[start_idx: start_idx + batch_size]
            encoded_vectors = model.encode([doc["contents"] for doc in batch], show_progress_bar=True)

            # combine the id, contents, and encoded vector for each document
            for doc, vec in zip(batch, encoded_vectors):
                embedding = {
                    "id": doc["id"],
                    "contents": doc["contents"],
                    "vector": vec.tolist() # convert numpy array to list
                }
                json.dump(embedding, f)
                f.write("\n")
    
def main():

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(args.model_path, device=device)

    data_path = os.path.join(args.data_root_path, args.data_format_type)
    print(f"Data path: {data_path}")

    output_path = os.path.join(args.output_root_path, f"sentenceBert-{args.data_format_type}")
    output_file = os.path.join(output_path, args.output_file_name)
    print(f"Output path: {output_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    documents = load_documents(data_path)
    print(f"Total documents: {len(documents)}")
    encode_documents(model, documents, output_file, args.batch_size)


if __name__ == "__main__":
    main()