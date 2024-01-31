import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--form", type=str, required=True, choices=["10K", "10Q"])
parser.add_argument("--encoder", type=str, required=True, choices=["dpr-ctx_encoder-multiset-base"]) 
parser.add_argument("--format_type", type=str, required=True, choices=["basic", "meta_data", "title"]) # basic, meta_data, title
parser.add_argument("--emb_file_name", type=str, default="embeddings")
parser.add_argument("--cik", type=str)
parser.add_argument("--start_year", type=int)
parser.add_argument("--end_year", type=int)
parser.add_argument("--item", type=str)

args = parser.parse_args()

def filter_embeddings(args, source_file, output_file):
    with open(source_file, 'r') as sourcefile, open(output_file, 'w') as outputfile:
        for line in sourcefile:
            data = json.loads(line)

            # Extract relevant parts from id
            # 20230227_10-K_797468_part1_item1a_para9
            id = data['id']
            id_components = id.split('_')
            year = int(id_components[0][:4])
            cik = id_components[2]
            item = id_components[4]

            # Check critieria
            match_cik = str(args.cik) == cik if args.cik else True
            match_year = (args.start_year <= year <= args.end_year) if args.start_year and args.end_year else True
            match_item = str(args.item) == item if args.item else True

            if match_cik and match_year and match_item:
                outputfile.write(line)

def generate_output_file(args):
    output_file = f"../embeddings/{args.form}/{args.encoder}-{args.format_type}"

    if args.cik:
        output_file += f"-cik{args.cik}"
    if args.start_year and args.end_year:
        output_file += f"-year{args.start_year}_{args.end_year}"
    if args.item:
        output_file += f"-item{args.item}"

    output_file += f"/{args.emb_file_name}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file

if __name__ == "__main__":
    source_file = f"../embeddings/{args.form}/{args.encoder}-{args.format_type}/{args.emb_file_name}.jsonl"
    output_file = generate_output_file(args)

    filter_embeddings(args, source_file, output_file)

    print(f"Filtered embeddings saved to {output_file}") 