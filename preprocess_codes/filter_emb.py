import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--form", type=str, required=True, choices=["10K", "10Q"])
parser.add_argument("--source_dir_name", type=str, required=True, help="the folder name of the source embeddings, (e.g., 'dpr-ctx_encoder-multiset-base-title')")
parser.add_argument("--emb_file_name", type=str, default="embeddings")
parser.add_argument("--cik", type=str)
parser.add_argument("--start_year", type=int)
parser.add_argument("--end_year", type=int)
parser.add_argument("--item", type=str)
parser.add_argument("--filter_out", action='store_true')  # flag to enable filter out

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
            match_year = (int(args.start_year) <= year <= int(args.end_year)) if args.start_year and args.end_year else True
            match_item = str(args.item) == item if args.item else True

            match_criteria = match_cik and match_year and match_item

            if match_criteria and not args.filter_out:      # write the lines that match the criteria if `filter_out` is not set (standard filtering)
                outputfile.write(line)
            elif not match_criteria and args.filter_out:    # write the lines that do not match the criteria if `filter_out` is set (exclude the lines that match the criteria)
                outputfile.write(line)

def generate_output_file(args):
    output_file = f"../embeddings/{args.form}/{args.source_dir_name}"

    if args.filter_out:
        output_file += "-filtered_out"

    if args.cik:
        output_file += f"-cik{args.cik}"
    if args.start_year and args.end_year:
        output_file += f"-year{args.start_year}_{args.end_year}"
    if args.item:
        output_file += f"-{args.item}"

    output_file += f"/{args.emb_file_name}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file

if __name__ == "__main__":
    source_file = f"../embeddings/{args.form}/{args.source_dir_name}/{args.emb_file_name}.jsonl"
    output_file = generate_output_file(args)

    filter_embeddings(args, source_file, output_file)

    print(f"Filtered embeddings saved to {output_file}") 