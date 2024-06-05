# Retriever pipeline in CFDA lab
The original codes are maintained under this directory: `/home/ybtu/FinNLP/`

Maintain the directory structure as defined in `config.py`. It should properly linked to the collections and indexes on my home directory. 

# Retriever Inference
Use `retrieve_paragraphs.py` for document retrieval. It supports sparse, dense, and hybrid retrieval models and works with various index types. The script is configurable via command-line arguments, making it versatile for different retrieval needs. 

## Example

```bash
python3 retrieve_paragraphs.py \
    --retriever_type dense \
    --index_type title \
    --prepend_info target_title \
    --cik 1045810 \
    --target_year 2022 \
    --target_item item7 \
    --k 10 \
    --target_paragraph para5 \
    --filter_name year2018_2022 \
    --post_filter \
    --output_jsonl_results \
    --d_encoder facebook/dpr-ctx_encoder-multiset-base \
    --q_encoder facebook/dpr-question_encoder-multiset-base
```

## Command-line Arguments

**Common Arguments**

* `--retriever_type`: Type of retriever to use (`dense`, `sparse`, `hybrid`)
* `--index_type`: Type of index formatting to use 
* `--prepend_info`: Type of information prepended to the target paragraph
* `--cik`: Central Index Key (CIK) of the target company
* `--target_year`: The target year of of financial reports that should be highlighted
* `--target_item`: Specific item in the target financial reports that should be highlighted
* `--k` (optional, default=10): Sets the number of top paragraphs to return
* `--target_paragraph` (optional): Specific paragraph to target. The retriever will process all the paragraph within the target_item if not provided
* `--filter_name` (optional): Specify the index with additional filter
* `--post_filter` (optional): Indicates whether to conduct post filtering to the retrieval results (e.g., exclude the paragraphs from the target company's item7)
* `--output_jsonl_results` (optional): enables the output of retrieval results into JSONL format to review the content of each paragraph

**Sparse Retriever Arguments**
* `--k1`: BM25 parameter k1 (default=1.5), used for adjusting document-term relevance 
* `--b`: BM25 parameter b (default=0.9), used for length normalization

**Dense Retriever Arguments**
* `--d_encoder`: Specifies the path to the document encoder checkpoint 
* `--q_encoder`: Specifies the path to the query encoder checkpoint

**Hybrid Retriever Arguments**

When utilizing the hybrid retriever, user should configure both the dense and sparse components. Users should specify the appropriate settings for the dense retriever, as previously detailed, and may also customize the sparse retriever settings if they wish to deviate from the defaults. 

* `--sparse_index_type` (optional, default="multi_fields"): Specifies the path to the document encoder checkpoint 
* `--sparse_filter_name` (optional): Filter name for the sparse index in the hybrid retriever. If not specified, it will use the same filter as the dense retriever
* `--weight_on_dense` (optional, default=False): Indicates whether to apply weights to the dense retrieval results
* `--alpha`(optional, default=0.1): Weight for blending results in the hybrid retriever.

# Retrieval Results Storage Structure
The retrieval results are stored under `/home/ybtu/FinNLP/retrieval_results_trec`.

## Subdirectory Structure
The results are categorized into various subdirectories, each following a specific naming convention to reflect the filtering criteria and data categorization:

1. CIK with Target Year:
It represents where the target paragraphs come from
* Format: `[cik]_[target_year]`
* Example: `200406_2020`
* In this examples, all the retrieval results under this directory are from 200406's 2020 financial report.

2. Index_type and Filtering Criteria:
* Format: `[index_type]-[filter_name]`
* Example: `title-year2018_2020`
* Breakdown: 
    * `index_type`: Indicates how the corpus that was indexed was formatted. (Refer to the corpus preprocessing section)
    * `filter_name`: The filtering criteria I have applied to construct the index. 

## File Naming
* Each file is named to represent the id of the target paragraph.
* The contents of each file consist of the top-k retrieval results related to that specific paragraph. 


# Building an Index on our Corpus
Codes related to preprocessing are maintained under the `preprocess_codes/` directory. 

To avoid duplicate files, the preprocessed results are maintained under the following directories.
* corpus: `/home/ybtu/FinNLP/collections`
* embeddings: `/home/ybtu/FinNLP/embeddings`
* index: `/home/ybtu/FinNLP/indexes`

## Corpus preprocessing
The `jsonl_to_formatted_data.py` script is designed to preprocess JSONL files processed by ythsiao for our retrieval systems. This preprocessing includes adding contextual information, filtering out irrelevant data, and reformatting content to meet specific retrieval requirements.  

### Usage

**Command Line Arguments**
* `--format_type`: Specifies the output format of the preprocessed data. Options include basic formatting, multi-field enrichment, metadata addition, etc.
* `start_index` and `--end_index`: Define the range of files to process, useful for batch processing or parallel execution
* `--max_length`: If set, truncates the content to a specified number of tokens, ensuring uniformity in document length

```bash
python3 jsonl_to_formatted_data.py \
        --format_type multi_fields \
        --start_index 0 \
        --end_index 100 \
        --max_length 512
```
This command preprocesses the first 100 JSONL files in the specified source directory, applying multi-field formatting to enrich the data and truncating content to a maximum of 512 tokens.

**Format Examples**

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
Example of title: `"Apple Inc. - Information Technology 2022 Q4 10-K Management's Discussion and Analysis of Financial Condition and Results of Operations; <paragraph_content>"`

## Encoding

Use `encode.sh` to encode documents with Dense encoders. For detailed instructions, visit the [Pyserini GitHub page](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-dense-vector-index)

## Filter

There are two Python scripts for filtering paragraphs and embeddings stored in JSONL format. These scripts allow you to apply inclusion or exclusion filters based on predefined criteria such as company CIK, filing year, and specific document items. Then, you can build the index on top of the filtered corpus or embeddings.

### Scripts
1. JSONL Document Filtering Script (`filter_collections.py`): Filters paragraphs from financial documents based on user-defined criteria
2. Embedding Filtering Script (`filter_emb.py`): Filters embeddings based on similar criteria

#### Filter JSONL Data
This script allows you to filter and extract specific paragraphs from JSONL files based on criteria such as SEC form type (e.g., 10K, 10Q), company CIK, specific years, and items within the filings. The script supports both inclusion and exclusion filtering modes to cater to diverse data preparation needs.

##### Command Line Arguments
* `--form`: Specifies the SEC form type. 
* `--format_type`: Specifies the formatting type of the data to process. 
Apply filter to exclude specific content in the embeddings to be indexed. 
* `--output_file_name`: Sets the base name for the output file. Defaults to `collections`
* `--cik`: Filters the data for a specific company's CIK number
* `--start_year` nad `--end_year`: Defines the year range for the documents to be included or excluded
* `--item`: Filters the paragraphs based on specific items in the filings
* `--filter_out`: If set, the script excludes the matching criteria instead of including them

### Naming convention
For a filter setting with setting format type `multi_fields`, including data from years `2018` to `2022`, the filtered collection would be stored under: 

`../collections/10K/multi_fields-year2018_2022/collections.jsonl`

## Build Index

* Use `build_sparse_index.sh` to build a BM25 index. 
* Use `build_dense_index_from_emb.sh` to build a Dense Vector Index

For detailed instructions on index construction, visit the [Pyserini GitHub page](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-dense-vector-index)