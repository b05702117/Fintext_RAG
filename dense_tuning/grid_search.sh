#!/bin/bash

d_encoder=$1 # facebook/dpr-ctx_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2
q_encoder=$2
index_type=$3
prepend_info=$4
filter_name="year2018_2022"
target_year="2022"
target_item="item7"
k=10
post_filter=true

declare -A cik_paragraphs
cik_paragraphs["1045810"]="para5"
cik_paragraphs["320193"]="para7"
cik_paragraphs["200406"]="para42"
cik_paragraphs["1585689"]="para74"
cik_paragraphs["1090727"]="para5"

echo "Running with index_type=$index_type, prepend_info=$prepend_info"
for cik in "${!cik_paragraphs[@]}"; do
    paragraph=${cik_paragraphs[$cik]}

    cmd="python3 dense_retrieval.py --d_encoder $d_encoder --q_encoder $q_encoder --index_type $index_type --prepend_info $prepend_info\
        --cik $cik --target_year $target_year --target_item $target_item --target_paragraph $paragraph\
        --k $k --filter_name $filter_name"

    if [ "$post_filter" = true ]; then
        echo "post filter!"
        cmd="$cmd --post_filter"
    fi

    eval $cmd
done