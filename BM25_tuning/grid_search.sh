#!/bin/bash

index_type=$1
prepend_info=$2
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


# Define ranges for k1 and b parameters
# k1_values=0.9
# k1_values=(0.0 0.5 0.9 1.0 1.5 2.0 2.5 3.0)
k1_values=(1.2)
b_values=(0.9)
# b_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# prepend_infos=("target_title" "target_company" "null")

# Loop over the parameter values
for k1 in "${k1_values[@]}"
do
    for b in "${b_values[@]}"
    do
        # for prepend_info in "${prepend_infos[@]}"
        # do
        echo "Running with k1=$k1, b=$b, prepend_info=$prepend_info"
        for cik in "${!cik_paragraphs[@]}"; do
            paragraph=${cik_paragraphs[$cik]}
            
            cmd="python3 sparse_retrieval.py --index_type $index_type --prepend_info $prepend_info --cik $cik\
                --target_year $target_year --target_item $target_item --k $k --target_paragraph $paragraph\
                --filter_name $filter_name --k1 $k1 --b $b"
            
            if [ "$post_filter" = true ]; then
                echo "post filter!"
                cmd="$cmd --post_filter"
            fi

            eval $cmd
        done
        # done
    done
done

echo "Grid search completed."
