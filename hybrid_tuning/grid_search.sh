#!/bin/bash

# hybrid
alpha=$1
weight_on_dense=false

# for sparse sercher
sparse_index_type="multi_fields"
k1=1.5
b=0.9

# for dense searcher
dense_index_type="title"
# d_encoder="sentence-transformers/all-mpnet-base-v2"
# q_encoder="sentence-transformers/all-mpnet-base-v2"
d_encoder="facebook/dpr-ctx_encoder-multiset-base"
q_encoder="facebook/dpr-question_encoder-multiset-base"

prepend_info="target_title"

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

for cik in "${!cik_paragraphs[@]}"; do
    paragraph=${cik_paragraphs[$cik]}

    cmd="python3 hybrid_retrieval.py --prepend_info $prepend_info --cik $cik --target_year $target_year --target_item $target_item\
         --target_paragraph $paragraph --filter_name $filter_name\
         --sparse_index_type $sparse_index_type --k1 $k1 --b $b\
         --dense_index_type $dense_index_type --d_encoder $d_encoder --q_encoder $q_encoder\
         --alpha $alpha"

    if [ "$post_filter" = true ]; then
        echo "post filter!"
        cmd="$cmd --post_filter"
    fi

    if [ "$weight_on_dense" = true ]; then
        echo "weight_on_dense!"
        cmd="$cmd --weight_on_dense"
    fi

    eval $cmd
done