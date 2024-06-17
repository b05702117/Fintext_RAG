import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True)
args = parser.parse_args()

def count_unique_query_ids(file_path):
    unique_query_ids = set()
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"No file found at {file_path}")
        return 0
    
    # Read the file and extract query IDs
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                query_id = parts[0]  # Assuming the query_id is the first element
                unique_query_ids.add(query_id)
    
    # Return the count of unique query IDs
    return len(unique_query_ids)


# Get the count of unique query IDs
num_unique_query_ids = count_unique_query_ids(args.file_path)
print(f"Number of unique query IDs: {num_unique_query_ids}")
