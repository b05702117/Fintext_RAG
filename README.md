# Retriever-Reader pipeline
The original codes are maintained under this directory: `/home/ybtu/FinNLP/`

# Important Note on Directory Structure
Maintain the directory structure as defined in `config.py`. It should properly linked to the collections and indexes on my home directory. 

# Retriever inference
Use `retrieve_paragraphs.py` for document retrieval. It supports both dense and sparse retrieval models and works with various index types. The script is configurable via command-line arguments, making it versatile for different retrieval needs. 

```
python retrieve_paragraphs.py \
    --model dense \
    --index_type title \
    --cik 1045810 \
    --target_year 2022 \
    --target_item item7 \
    --k 10 \
    --target_paragraph para5 \
    --filter_name title-year2018_2022-filtered_out-cik1045810-year2022_2022-item7 \
    --output_jsonl_results
```

Command-line Arguments
* `--model`: Type of model to use for retrieval (`dense` or `sparse`)
* `--index_type`: Type of index to use 
* `--cik`: Central Index Key (CIK) of the target company
* `--target_year`: The target year of of financial reports that should be highlighted
* `--target_item`: Specific item in the target financial reports that should be highlighted
* `--k` (optional, default=10): Sets the number of top paragraphs to return
* `--target_paragraph` (optional): Specific paragraph to target. The retriever will process all the paragraph within the target_item if not provided
* `--filter_name` (optional): Specify the index with additional filter
* `--output_jsonl_results` (optional): enables the output of retrieval results into JSONL format to review the content of each paragraph

# Corpus, Embeddings, and Indexes
To avoid duplicate files, they are all maintained under my directory.

## Corpus
Go to this path to review my current corpus collection:
`/home/ybtu/FinNLP/collections`

The structure of the original document is maintained as follows:
```
{
    "id": "<paragraph_id>", 
    "content": "<paragraph_content>"
}
```

To test the effect of concatenating various information to the original document, I have established several formats for encoding. 

* `basic`: This format focuses solely on the content of the paragraph 
```
{
    "id": "<paragraph_id>", 
    "content": "<paragraph_content>"
}
```

* `meta_data`: In this format, metadata is prepended to the paragraph, enriching the context for retrieval. 
```
{
    "id": "<paragraph_id>", 
    "content": "<meta_data>; <paragraph_content>"
}
```
Example of meta_data: `cik: 320193 company_name: Apple Inc., filing_date: 20221028, form: 10-K; <paragraph_content>`

* `title`: This format prepends the title to the paragraph content, aiding in retrieval scenarios where the title's context is crucial.
```
{
    "id": "<paragraph_id>", 
    "content": "<title>; <paragraph_content>"
}
```
Example of title: `"Apple Inc. 2022 Q4 10-K MD&A; <paragraph_content>"`

## Embedding
Currently, I employ the `dpr-ctx_encoder-multiset-base` as both the document and query encoder. 

### Filter
Apply filter to exclude specific content in the embeddings to be indexed. 

Review the following path to see what kinds of filters I have applied. 

```/home/ybtu/FinNLP/indexes```

# Retrieval Results Storage Structure
The retrieval results are stored under `/home/ybtu/FinNLP/retrieval_results_trec`.

## Root Directory
`/home/ybtu/FinNLP/retrieval_results_trec`: This is the base directory where all retrieval results are organized. 

## Subdirectory Structure
Within the root directory, the results are categorized into various subdirectories, each following a specific naming convention to reflect the filtering criteria and data categorization:

1. CIK with Target Year:
It represents where the target paragraphs come from
* Format: `[cik]_[target_year]`
* Example: `200406_2020`
* In this examples, all the retrieval results under this directory are from 200406's 2020 financial report.

2. Index_type and Filtering Criteria:
* Format: `[index_type]-[filter_name]`
* Example: `title-year2018_2020-filtered_out-cik200406-year2020_2020-item7`
* Breakdown: 
    * `index_type`: Indicates how the corpus that was indexed was formatted. (Refer to the corpus section)
    * `filter_name`: The filtering criteria I have applied to construct the index. 

## File Naming
* Each file is named to represent the id of the target paragraph.
* The contents of each file consist of the top-k retrieval results related to that specific paragraph. 

## Current filter applied to dense retrieval:
* Suppose the target paragraphs are from 200406's 2020 10-K item7. The retrieval corpus should exclude paragraphs from its item7 in 2020. 
* Applied filter:
    1. year between 2018 - 2020
    2. exclude cik=='200406' and year=='2020' and item == 'item7'