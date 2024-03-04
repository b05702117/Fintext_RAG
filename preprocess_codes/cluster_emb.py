import os
import json
import argparse
# from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--form", type=str, required=True, choices=["10K", "10Q"])
parser.add_argument("--source_dir_name", type=str, required=True) # the folder name of the source embeddings, e.g. "dpr-ctx_encoder-multiset-base-title"
parser.add_argument("--emb_file_name", type=str, default="embeddings")

args = parser.parse_args()

with open('cik_to_sector.json', 'r') as f:
    cik_to_sector = json.load(f)

def group_embeddings_by_sector(source_file):
    clustered_embs = {}

    sectors = set(cik_to_sector.values())
    for sector in sectors:
        clustered_embs[sector] = []
        print(sector)

    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            id = data['id']
            id_components = id.split('_')
            cik = id_components[2]

            clustered_embs[cik_to_sector[cik]].append(data)
    
    return clustered_embs

def generate_output_file(args, clustered_embs):
    sectors = set(cik_to_sector.values())

    output_file = f"../embeddings/{args.form}/{args.source_dir_name}"

    for sector in sectors:
        print("Generating output file for", sector)
        embeddings = clustered_embs[sector]
        sector_output_file = output_file + f"-{sector.replace(' ', '_')}/{args.emb_file_name}.jsonl"
        os.makedirs(os.path.dirname(sector_output_file), exist_ok=True)
        
        with open(sector_output_file, 'w') as f:
            for emb in embeddings:
                f.write(json.dumps(emb) + '\n')

if __name__ == "__main__":
    source_file = f"../embeddings/{args.form}/{args.source_dir_name}/{args.emb_file_name}.jsonl"
    clustered_embs = group_embeddings_by_sector(source_file)
    generate_output_file(args, clustered_embs)