import os
import argparse
import fnmatch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cik", type=str, required=True)
    parser.add_argument("--target_year", type=int, default=2022)
    parser.add_argument("--target_item", type=str, default="item7")
    parser.add_argument("--target_paragraph", type=str, required=True)

    args = parser.parse_args()

    search_pattern = f"*_{args.target_item}_{args.target_paragraph}.txt"
    target_file_dir = f"{args.cik}_{args.target_year}"
    ignore_file = 'retrieval_results.txt'  # File to ignore

    output_file = f"{args.cik}_{args.target_year}_{args.target_item}_{args.target_paragraph}.txt"

    with open(output_file, "w") as output:
        for retrieval_sys in os.listdir(target_file_dir):
            if os.path.isdir(f"{target_file_dir}/{retrieval_sys}"):
                for file_name in os.listdir(f"{target_file_dir}/{retrieval_sys}"):
                    if fnmatch.fnmatch(file_name, search_pattern):
                        with open(f"{target_file_dir}/{retrieval_sys}/{file_name}", "r") as f:
                            output.write(f.read())

if __name__ == '__main__':
    main()