import os
import argparse

def combine_results(base_dir, output_dir, retrieval_tag):
    os.makedirs(output_dir, exist_ok=True)

    combined_file_path = os.path.join(output_dir, f"{retrieval_tag}.txt")

    with open(combined_file_path, 'w') as output_file:
        # walk through the directory structure
        for root, dirs, files in os.walk(base_dir):
            # check if the current directory matches the retrieval tag
            if retrieval_tag in root:
                # iterate over the files in the current directory
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        # read the contents of the file and write to the combined file
                        with open(file_path, 'r') as f:
                            output_file.write(f.read())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_tag", type=str, required=True)
    args = parser.parse_args()

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    base_dir = script_directory
    output_dir = os.path.join(script_directory, "combined_results")

    combine_results(base_dir, output_dir, args.retrieval_tag)

if __name__ == "__main__":
    main()