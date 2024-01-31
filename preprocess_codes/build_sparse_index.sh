# /bin/bash

FORM=$1                 # 10Q or 10K
SOURCE=$2               # basic, meta_data, title
INDEX_NAME=${3}

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../collections/${FORM}/${SOURCE} \
  --index ../indexes/${FORM}/${INDEX_NAME} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw