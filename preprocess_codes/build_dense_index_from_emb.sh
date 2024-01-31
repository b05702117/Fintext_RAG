# /bin/bash

FORM=$1                 # 10Q or 10K
ENCODER=$2              # dpr-ctx_encoder-multiset-base
FORMAT_TYPE=$3          # basic, meta_data, title
FILTER_NAME=${4:-null}  # year2018_2019

source_dir=../embeddings/${FORM}/${ENCODER}-${FORMAT_TYPE}
if [ $FILTER_NAME != "null" ]; then
    source_dir=${source_dir}-${FILTER_NAME}
fi
index_dir=../indexes/${FORM}/${ENCODER}-${FORMAT_TYPE}
if [ $FILTER_NAME != "null" ]; then
    index_dir=${index_dir}-${FILTER_NAME}
fi

python -m pyserini.index.faiss \
    --input ${source_dir} \
    --output ${index_dir}