import os
import argparse
import fnmatch

def cluster_and_output(file_path):
    clusters = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                tag = parts[5]
                if tag not in clusters:
                    clusters[tag] = []
                clusters[tag].append(line)
    
    for tag, lines in clusters.items():
        output_file_name = f"results_{tag}.txt"
        with open(output_file_name, 'w') as f:
            for line in lines:
                f.write(line)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, required=True)

    args = parser.parse_args()

    cluster_and_output(args.file_path)

if __name__ == "__main__":
    main()

    