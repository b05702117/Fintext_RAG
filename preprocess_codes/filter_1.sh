#!/bin/bash

form=$1
target_year=$2
target_cik=$3

source_dir_name="dpr-ctx_encoder-multiset-base-title"
emb_file_name="embeddings"
from_year=2018
to_year=$target_year

# filter year range
year_filtered_dir_name="${source_dir_name}-year${from_year}_${to_year}"
if [ ! -d "../embeddings/${form}/${year_filtered_dir_name}" ]; then
    echo "The filter for year${from_year}_${to_year} does not exist"
    python filter_emb.py --form $form --source_dir_name $source_dir_name --emb_file_name $emb_file_name \
        --start_year $from_year --end_year $to_year
fi

# exclude target cik's item7 in target year
echo "Excluding ${target_cik}'s item7 in year${target_year}"
python filter_emb.py --form $form --source_dir_name $year_filtered_dir_name --emb_file_name $emb_file_name \
    --cik $target_cik --start_year $target_year --end_year $target_year --item item7 --filter_out

