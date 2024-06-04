import utils # utils.py
import os
import json
import torch
from typing import List, Dict
from pyserini.search.faiss import FaissSearcher, DenseSearchResult
from config import FORMMATED_DIR
from transformers import DPRReader, DPRReaderTokenizer, DPRQuestionEncoderTokenizer
from pyserini.search.faiss import QueryEncoder
from pyserini.search.hybrid import HybridSearcher
from cnc_highlighting.encode import BertForHighlightPrediction
from sentence_transformers import SentenceTransformer

K = 10 # top k documents to retrieve

class CustomHybridSearcher(HybridSearcher):
    '''
    CustomHybridSearcher inherits from HybridSearcher and overrides the search method to include the fields parameter.
    '''
    def __init__(self, dense_searcher, sparse_searcher):
        super().__init__(dense_searcher, sparse_searcher)

    def search(self, query: str, k0: int = 10, k: int = 10, alpha: float = 0.1, normalization: bool = False, weight_on_dense: bool = False, fields: dict = None) -> List[DenseSearchResult]:
        dense_hits = self.dense_searcher.search(query, k0)
        sparse_hits = self.sparse_searcher.search(query, k0, fields=fields)
        return self._hybrid_results(dense_hits, sparse_hits, alpha, k, normalization, weight_on_dense)
    
    def batch_search(self, queries: List[str], q_ids: List[str], k0: int = 10, k: int = 10, threads: int = 1,
            alpha: float = 0.1, normalization: bool = False, weight_on_dense: bool = False, fields: dict = None) \
            -> Dict[str, List[DenseSearchResult]]:
        dense_result = self.dense_searcher.batch_search(queries, q_ids, k0, threads)
        sparse_result = self.sparse_searcher.batch_search(queries, q_ids, k0, threads, fields=fields)
        hybrid_result = {
            key: self._hybrid_results(dense_result[key], sparse_result[key], alpha, k, normalization, weight_on_dense)
            for key in dense_result
        }
        return hybrid_result

class CustomSentenceTransformerEncoder(QueryEncoder):
    def __init__(self, model_path: str, encoded_query_dir: str = None, device: str = 'cpu'):
        super().__init__(encoded_query_dir)

        if model_path:
            self.model = SentenceTransformer(model_path)
            self.tokenizer = self.model.tokenizer
            self.device = device
            self.model.to(self.device)
            self.has_model = True
        else:
            if not self.has_encoded_query:
                raise ValueError('Either model_path or encoded_query_dir must be provided.')
    
    def encode(self, query: str):
        if self.has_model:
            embeddings = self.model.encode(query, convert_to_tensor=True)
            return embeddings
        else:
            return super().encode(query)
        
class BaseRetriever:
    ''' Base class for document retrievers '''
    def __init__(self, searcher, docs_dir=FORMMATED_DIR, k=K):
        self.searcher = searcher
        self.docs_dir = docs_dir
        self.k = k
    
    def search_documents(self, query, filter_function=None):
        m = 2
        filtered_hits = []
        while len(filtered_hits) < self.k:
            hits = self.searcher.search(query, k=self.k*m)
            if filter_function:
                hits = filter_function(hits)
            else:
                hits = hits[:self.k]
            filtered_hits = hits
            m += 1
        return filtered_hits[:self.k]

class SparseDocumentRetriever(BaseRetriever):
    def __init__(self, searcher, fields=None, docs_dir=FORMMATED_DIR, k=K):
        super().__init__(searcher, docs_dir, k)
        self.fields = fields
    
    def search_documents(self, query, filter_function=None):
        m = 2
        filtered_hits = []
        while len(filtered_hits) < self.k:
            hits = self.searcher.search(query, k=self.k*m, fields=self.fields)
            if filter_function:
                hits = filter_function(hits)
            else:
                hits = hits[:self.k]
            filtered_hits = hits
            m += 1
        return filtered_hits[:self.k]

class DenseDocumentRetriever(BaseRetriever):
    def __init__(self, searcher, docs_dir=FORMMATED_DIR, k=K):
        super().__init__(searcher, docs_dir, k)
        self.q_tokenizer = searcher.query_encoder.tokenizer
    
    def truncate_query(self, query, max_length=512):
        ''' truncate the query to the maximum length '''
        tokens = self.q_tokenizer(query, return_tensors='pt', truncation=True, max_length=max_length)
        truncated_query = self.q_tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        return truncated_query

    def search_documents(self, query, filter_function=None):
        truncated_query = self.truncate_query(query)
        return super().search_documents(truncated_query, filter_function)

class HybridDocumentRetriever(BaseRetriever):
    def __init__(self, searcher, docs_dir=FORMMATED_DIR, k=K, alpha=0.1, weight_on_dense=False, fields=None):
        super().__init__(searcher, docs_dir, k)
        self.alpha = alpha
        self.weight_on_dense = weight_on_dense
        self.fields = fields
    
    def search_documents(self, query, filter_function=None):
        m = 2
        filtered_hits = []
        while len(filtered_hits) < self.k:
            hits = self.searcher.search(
                query, 
                k0=self.k*m, 
                k=self.k*m, 
                weight_on_dense=self.weight_on_dense, 
                alpha=self.alpha, 
                fields=self.fields
            )
            if filter_function:
                hits = filter_function(hits)
            else:
                hits = hits[:self.k]
            filtered_hits = hits
            m += 1
        return filtered_hits[:self.k]

class DprHighlighter:
    ''' 
    The DprHighlighter serves as the baseline to compare with generator 
    https://huggingface.co/facebook/dpr-reader-multiset-base
    '''
    def __init__(self, model_name: str = 'facebook/dpr-reader-multiset-base', tokenizer_name: str = 'facebook/dpr-reader-multiset-base', device: str = 'cpu'):
        self.device = device
        self.model = DPRReader.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = DPRReaderTokenizer.from_pretrained(tokenizer_name)

    @staticmethod
    def find_max_idx(logits, dim=-1):
        probs = torch.softmax(logits, dim=dim)
        return torch.argmax(probs)
    
    @staticmethod
    def mask_prior_logits(logits, mask_idx):
        '''
        Masks the logits by setting the value before the mask_idx to -inf
            logits: tensor of shape (batch_size, sequence_length)
            mask_idx: tensor of shape (batch_size, )
        '''
        # Clone the logits to avoid modifying the original tensor
        masked_logits = logits.clone()

        sequence_range = torch.arange(masked_logits.shape[1])
        mask = sequence_range[None, :] < mask_idx[:, None]
        masked_logits[mask] = -float("inf")

        return masked_logits

    def extract_answer_span(self, token_ids, start_position, end_position):
        answer_tokens = token_ids[start_position : end_position + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer

    def find_target_start_position(self, input_ids):
        ''' 
            encoded_inputs: [CLS] <reference> [SEP] <title> [SEP] <target_paragraph> [SEP]
            input_ids: tensor of shape (batch_size, sequence_length)
        '''
        sep_token_id = self.tokenizer.sep_token_id
        # Create a boolean mask where positions of SEP tokens are True
        sep_mask = input_ids == sep_token_id
        # Find the indices of all SEP tokens using the mask
        sep_positions = torch.nonzero(sep_mask, as_tuple=False) 

        # Initialize a tensor to hold the second SEP token index for each sequence
        batch_size = input_ids.shape[0]
        second_sep_positions = torch.zeros(batch_size, dtype=torch.int64)

        # Loop through the sequences in the batch
        for i in range(batch_size):
            # Find SEP positions for the current sequence
            sep_indices = sep_positions[sep_positions[:, 0] == i, 1]
            # Store the second SEP position for the current sequence
            second_sep_positions[i] = sep_indices[1]
        
        return second_sep_positions + 1


    def highlighting_outputs(self, target, target_title, references):
        ''' 
        target: the target paragraph that should be highlighted
        targe_title: the title of the target paragraph
        references: the retrieved paragrpah, which are the reference for our highlighting work
        DPRReaderTokenizer output: [CLS] <questions> [SEP] <titles> [SEP] <texts>
        BERT input: [CLS] <texts> [SEP] <questions> 
        TODO: 
            - 限制start_logits & end_logits在最後面? (作為好多個reference的最終highlight)
            - handle paragraph that is too long
        '''
        
        targets = [target] * len(references)        # our target paragraph that should be highlighted
        titles = [target_title] * len(references)   # the title of the target paragraph

        encoded_inputs = self.tokenizer(
            questions=references,       # retrieved documents are the reference for our highlighting work
            titles=titles,              
            texts=targets,
            padding=True if len(targets) > 1 else False,
            return_tensors="pt", 
            truncation=True,            # TODO: handle paragraph that is too long
            max_length=512
        )
        
        outputs = self.model(**encoded_inputs)

        return encoded_inputs, outputs
    
    def output_highlighting_results(self, output_file, results):
        with open(os.path.join('highlighting_results', output_file), 'w') as f:
            json.dump(results, f)

    def visualize_highlight_span(self, encoded_inputs, ref_ids, relevance_logits, start_logits, end_logits, output_file_path=None):
        results = []
        num_ref = start_logits.shape[0]

        # Sort the relevance logits in descending order
        relevance_probs = torch.softmax(relevance_logits, dim=-1)
        sorted_indices = torch.argsort(relevance_probs, descending=True)

        for i in sorted_indices:

            start_idx = self.find_max_idx(start_logits[i])
            end_idx = self.find_max_idx(end_logits[i])
            highlighted_span = self.extract_answer_span( 
                encoded_inputs['input_ids'][i],
                start_idx,
                end_idx
            )
            
            result = {
                "relevance_prob": f"{relevance_probs[i]:.4f}",
                "highlighted_span": highlighted_span, 
                "reference_id": ref_ids[i],
                "reference_paragraph": utils.retrieve_paragraph_from_docid(ref_ids[i])
            }
            results.append(result)

            print(f"{relevance_probs[i]:.4f} reference {ref_ids[i]}:")
            print(f"start_idx: {start_idx}, end_idx: {end_idx}, span: {highlighted_span}")

        if output_file_path:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                for result in results:
                    json.dump(result, f)
                    f.write('\n')

class CncBertHighlighter:
    def __init__(self, model_name: str = 'DylanJHJ/bert-base-final-v0-ep2', device: str = 'cpu'):
        self.device = device
        self.model = BertForHighlightPrediction.from_pretrained(model_name)
        self.model.to(self.device)
    
    def highlighting_outputs(self, target, text_references):
        num_references = len(text_references)
        targets = [target] * num_references

        outputs = self.model.encode(
            text_tgt=targets,
            text_ref=text_references,
            pretokenized=False, 
            return_reference=False
        )

        return outputs
    
    def find_highest_prob_word(self, words_tgt, word_probs_tgt, n):
        sorted_indices = word_probs_tgt.argsort()[::-1]  # Sort indices in descending order
        top_n_indices = sorted_indices[:n]  # Get the top-n indices
        top_n_words = [words_tgt[i] for i in top_n_indices]
        return top_n_words
    
    def visualize_top_k_highlight(self, highlight_results, highlight_words_cnt=5):
        for i in range(len(highlight_results)):
            words_tgt = highlight_results[i]['words_tgt']
            word_probs_tgt = highlight_results[i]['word_probs_tgt']
            top_k_words = self.find_highest_prob_word(words_tgt, word_probs_tgt, highlight_words_cnt)
            print(f"reference {i+1}:", top_k_words)
            # print(f"reference {i+1}:")
            # print(top_k_words)

class DprReader:
    ''' https://huggingface.co/facebook/dpr-reader-multiset-base '''
    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu'):
        self.device = device
        self.model = DPRReader.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = DPRReaderTokenizer.from_pretrained(tokenizer_name or model_name)
    
    @staticmethod
    def find_max_idx(logits, dim=-1):
        probs = torch.softmax(logits, dim=dim)
        return torch.argmax(probs)

    @staticmethod
    def extract_answer_span(tokenizer, token_ids, start_position, end_position):
        answer_tokens = token_ids[start_position : end_position + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer

    def generate_model_outputs(self, query, titles, texts):
        questions = [query] * len(titles)

        encoded_inputs = self.tokenizer(
            questions=questions, 
            titles=titles,  
            texts=texts,
            padding=True if len(titles) > 1 else False,
            return_tensors="pt"
        )

        outputs = self.model(**encoded_inputs)

        return encoded_inputs, outputs
    
    def visualize_answer_span(self, encoded_inputs, ref_ids, relevance_logits, start_logits, end_logits):
        num_ref = len(ref_ids)

        # Sort the relevance logits in descending order
        relevance_probs = torch.softmax(relevance_logits, dim=-1)
        sorted_indices = torch.argsort(relevance_probs, descending=True)

        for i in sorted_indices:        
            start_idx = DprReader.find_max_idx(start_logits[i])
            end_idx = DprReader.find_max_idx(end_logits[i])
            highlighted_span = DprReader.extract_answer_span( # TODO: 確認為什麼這邊寫self.不行
                self.tokenizer,
                encoded_inputs['input_ids'][i],
                start_idx,
                end_idx
            )
            
            print(f"{relevance_probs[i]:.4f} reference {ref_ids[i]}:")
            print(f"start_idx: {start_idx}, end_idx: {end_idx}, span: {highlighted_span}")

class QAPipeline:
    ''' TODO '''
    def __init__(self, retriever, reader):
        self.retriever = retriever
        self.reader = reader
    
    def answer_question(self, query):
        titles, texts = self.retriever.retrieve_and_process_documents(query)
        return self.reader.find_answer(query, titles, texts)

class HighlightPipeline:
    ''' TODO '''
    def __init__(self, retriever, highlighter):
        self.retriever = retriever
        self.highlighter = highlighter

    def highlight_span(self, target):
        titles, texts = self.retriever.retrieve_and_process_documents(target)
        pass

    @staticmethod
    def filter_retrived_documents():
        pass

def generate_statistics_summary(titles):
    # Count the number of documents retrieved
    statistics_summary = {'year': dict(), 'form': dict(), 'cik': dict(), 'part': dict(), 'item': dict()}

    for title in titles:
        title_parts = title.split('_')
        year, form, cik, part, item = title_parts[0][:4], title_parts[1], title_parts[2], title_parts[3], title_parts[4]

        for category, value in zip(['year', 'form', 'cik', 'part', 'item'], [year, form, cik, part, item]):
            statistics_summary[category].setdefault(value, 0)
            statistics_summary[category][value] += 1

    return statistics_summary

def print_hits(hits, display_top_n=10):
    for i in range(display_top_n):
        print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
        print(utils.retrieve_paragraph_from_docid(hits[i].docid))
    print()

def output_hits(hits, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for hit in hits:
            result = {
                'id': hit.docid,
                'score': float(hit.score), # convert float32 to standard float
                'contents': utils.retrieve_paragraph_from_docid(hit.docid)
            }
            json.dump(result, f)
            f.write('\n')