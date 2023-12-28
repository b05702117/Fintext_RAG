# /bin/bash

python -m pyserini.encode \
  input   --corpus ../collections/basic \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ../indexes/dpr-ctx_encoder-multiset-base \
          --to-faiss \
  encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
          --batch 32 \
          --fp16  # if inference with autocast()
