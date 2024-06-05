#!/bin/bash

# Define the parent directory
PARENT_DIR="$(dirname "$(realpath "$0")")" # retrieval_results_trec/

for cik_year_folder in "$PARENT_DIR"/*; do
    if [ -d "$cik_year_folder" ]; then
        echo "Processing folder: $cik_year_folder"
        # Initialize or clear the retrieval_results.txt file
        RETRIEVAL_FILE="$cik_year_folder/retrieval_results.txt"
        > "$RETRIEVAL_FILE"

        for sub_dir in "$cik_year_folder"/*; do
            if [ -d "$sub_dir" ]; then
                # echo "Processing sub-folder: $sub_dir"

                # # Loop through each text file in the sub-folder
                for file in "$sub_dir"/*.txt; do
                    if [ -f "$file" ]; then
                        # echo "Processing file: $file"
                        while IFS= read -r line; do
                            echo "$line" >> "$RETRIEVAL_FILE"
                        done < "$file"
                    fi
                done
            fi
        done
    fi
done

echo "Retrieval process completed."
