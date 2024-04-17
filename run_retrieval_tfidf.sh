#!/bin/bash

target_year="2022"
target_item="item7"
k=10
self_ref=false

declare -A cik_paragraphs
cik_paragraphs["320193"]="para7"
cik_paragraphs["1045810"]="para5"
cik_paragraphs["200406"]="para42"
cik_paragraphs["1585689"]="para74"
cik_paragraphs["1090727"]="para5"


for cik in "${!cik_paragraphs[@]}"; do
    paragraph=${cik_paragraphs[$cik]}

    cmd="python3 tf_idf_searcher.py --cik $cik --target_year $target_year \
        --target_item $target_item --target_paragraph $paragraph --k $k \
        --post_filter --output_jsonl_results"

    if [ "$self_ref" = true ]; then
        cmd="$cmd --self_reference"
    fi

    # Execute the command
    eval $cmd
done