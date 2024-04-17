#!/bin/bash

model=$1
index_type=$2
d_encoder=${3:-""}
q_encoder=${4:-""}
filter_name="year2018_2022"
target_year="2022"
target_item="item7"
k=10
post_filter=true
output_jsonl_results=true
hybrid_sparse_filter=false # filter cik using sparse retrieval

declare -A cik_paragraphs
cik_paragraphs["320193"]="para7"
cik_paragraphs["1045810"]="para5"
cik_paragraphs["200406"]="para42"
cik_paragraphs["1585689"]="para74"
cik_paragraphs["1090727"]="para5"


for cik in "${!cik_paragraphs[@]}"; do
    paragraph=${cik_paragraphs[$cik]}

    # filter_name_cik=$filter_name-cik$cik

    cmd="python3 retrieve_paragraphs.py --model $model --index_type $index_type --cik $cik --target_year $target_year\
        --target_item $target_item --k $k --target_paragraph $paragraph \
        --filter_name $filter_name"
    
    # Add d_encoder and q_encoder to command if they are not empty
    [ ! -z "$d_encoder" ] && cmd="$cmd --d_encoder $d_encoder"
    [ ! -z "$q_encoder" ] && cmd="$cmd --q_encoder $q_encoder"
    # [ ! -z "$hybrid_sparse_filter" ] && cmd="$cmd --hybrid_sparse_filter $hybrid_sparse_filter"

    if [ "$post_filter" = true ]; then
        cmd="$cmd --post_filter"
    fi

    if [ "$output_jsonl_results" = true ]; then
        cmd="$cmd --output_jsonl_results"
    fi

    if [ "$hybrid_sparse_filter" = true ]; then
        cmd="$cmd --hybrid_sparse_filter $filter_name-cik$cik"
    else
        cmd="$cmd --hybrid_sparse_filter $filter_name"
    fi

    # Execute the command
    eval $cmd
done