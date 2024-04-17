import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--form", type=str, required=True, choices=["10K", "10Q"])
parser.add_argument("--format_type", type=str, required=True, choices=["basic", "company_name", "multi_fields", "title"])
parser.add_argument("--output_file_name", type=str, default="collections")
parser.add_argument("--cik", type=str)
parser.add_argument("--start_year", type=int)
parser.add_argument("--end_year", type=int)
parser.add_argument("--item", type=str)
parser.add_argument("--filter_out", action='store_true')  # flag to enable filter out

args = parser.parse_args()

def filter_paragraph(args, source_file, output_file):
    with open(source_file, 'r') as sourcefile, open(output_file, 'a') as outputfile:
        for line in sourcefile:
            data = json.loads(line)

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
    output_file = f"../collections/{args.form}/{args.format_type}"

    if args.filter_out:
        output_file += "-filtered_out"
    
    if args.start_year and args.end_year:
        output_file += f"-year{args.start_year}_{args.end_year}"
    if args.cik:
        output_file += f"-cik{args.cik}"
    if args.item:
        output_file += f"-{args.item}"

    output_file += f"/{args.output_file_name}.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Clear the content of the output file at the beginning of the execution
    with open(output_file, 'w') as file:
        file.truncate(0)

    return output_file

if __name__ == "__main__":
    source_dir = f"../collections/{args.form}/{args.format_type}"
    output_file = generate_output_file(args)
    for file_name in os.listdir(source_dir):
        filter_paragraph(args, f"{source_dir}/{file_name}", output_file)
