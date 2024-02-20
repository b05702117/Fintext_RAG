# /bin/bash

FORM=$1                 # 10Q or 10K
SOURCE=$2               # basic, meta_data, title, ner, ner_concat
ENCODER_NAME=$3         # facebook/dpr-ctx_encoder-multiset-base; sentence-transformers/all-mpnet-base-v2; sentence-transformers/multi-qa-mpnet-base-dot-v1
DEVICE_INDEX=${4:-0}    # Device index (e.g., 0 for 'cuda:0')
BATCH_SIZE=${5:-32}

# # Replace '/' with '_' and remove any potential spaces
# ENCODER_FILENAME=$(echo "${ENCODER_NAME}" | sed 's/[\/ ]/_/g')
# echo "Encoder filename: ${ENCODER_FILENAME}"

# Extract the part after the last slash
ENCODER_FILENAME="${ENCODER_NAME##*/}"
echo "Encoder filename: ${ENCODER_FILENAME}"

device="cuda:${DEVICE_INDEX}"

python -m pyserini.encode \
  input   --corpus ../collections/${FORM}/${SOURCE} \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ../embeddings/${FORM}/${ENCODER_FILENAME}-${SOURCE} \
  encoder --encoder ${ENCODER_NAME} \
          --batch ${BATCH_SIZE} \
          --device ${device} \
          --fp16  # if inference with autocast()