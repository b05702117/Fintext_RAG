# /bin/bash

FORM=$1                 # 10Q or 10K
FORMAT_TYPE=$2          # multi_fields
FILTER_NAME=${3:-null}  # year2018_2019

source_dir=../collections/${FORM}/${FORMAT_TYPE}
if [ $FILTER_NAME != "null" ]; then
    source_dir=${source_dir}-${FILTER_NAME}
fi

index_dir=../indexes/${FORM}/${FORMAT_TYPE}
if [ $FILTER_NAME != "null" ]; then
    index_dir=${index_dir}-${FILTER_NAME}
fi

# Check if the directory exists, if not, create it
if [ ! -d "$index_dir" ]; then
    mkdir -p $index_dir
fi

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${source_dir} \
  --index ${index_dir} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw