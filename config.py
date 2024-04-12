# config.py

import os
import json

ROOT = "/home/ybtu/FinNLP"
RAW_DIR = "/home/ybtu/FinNLP/Jsonl_Data"

# for 10-K
FORMMATED_DIR = "/home/ybtu/FinNLP/collections/10K/basic"
EMB_DIR = "/home/ybtu/FinNLP/collections/10K/emb"
INDEX_DIR = "/home/ybtu/FinNLP/indexes/10K"

with open(os.path.join(ROOT, 'collections', 'cik_to_company.json'), 'r') as f:
    CIK_TO_COMPANY = json.load(f)

with open(os.path.join(ROOT, 'collections', 'item_mapping.json'), 'r') as f:
    ITEM_MAPPING = json.load(f)

with open(os.path.join(ROOT, 'collections', 'cik_to_sector.json'), 'r') as f:
    CIK_TO_SECTOR = json.load(f)