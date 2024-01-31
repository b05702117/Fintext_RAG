# /bin/bash

FORM=$1                 # 10Q or 10K
SOURCE=$2               # basic, meta_data, title
BATCH_SIZE=${3:-32}

# python -m pyserini.encode \
#   input   --corpus ../collections/${FORM}/${SOURCE} \
#           --shard-id 0 \
#           --shard-num 1 \
#   output  --embeddings ../indexes/${FORM}/dpr-ctx_encoder-multiset-base-${SOURCE} \
#           --to-faiss \
#   encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
#           --batch ${BATCH_SIZE} \
#           --fp16  # if inference with autocast()

python -m pyserini.encode \
  input   --corpus ../collections/${FORM}/${SOURCE} \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ../embeddings/${FORM}/dpr-ctx_encoder-multiset-base-${SOURCE} \
  encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
          --batch ${BATCH_SIZE} \
          --fp16  # if inference with autocast()