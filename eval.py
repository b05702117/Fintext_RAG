import json
import os
import csv
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize

from bert_score import BERTScorer
scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

from utils import get_10K_file_name, convert_docid_to_title, retrieve_paragraph_from_docid

def cosine_similarity_tfidf(para1: str, para2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([para1, para2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def jaccard_similarity(para1: str, para2: str) -> float:
    words_para1 = set(para1.split())
    words_para2 = set(para2.split())
    intersection = words_para1.intersection(words_para2)
    union = words_para1.union(words_para2)
    if not union:
        return 1.0
    return len(intersection) / len(union)

def semantic_similarity_bert(para1: str, para2: str, model):
    emb_1 = model.encode(para1, convert_to_tensor=True)
    emb_2 = model.encode(para2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb_1, emb_2)

    return cosine_scores.item()

def calculate_bleu_score(ref, cand):
    smoothie = SmoothingFunction().method4
    ref_tokens = word_tokenize(ref)
    cand_tokens = word_tokenize(cand)
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

def calculate_bert_score(ref, cand):
    P, R, F1 = scorer.score([cand], [ref])
    return P, R, F1

def parse_trec_file(file):
    results = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, rank, score, retrieval_type = parts
            if query_id not in results:
                results[query_id] = {}
            if retrieval_type not in results[query_id]:
                results[query_id][retrieval_type] = []
            results[query_id][retrieval_type].append((doc_id, int(rank)))  # Store doc_id and rank as a tuple
    # Sort the doc_id based on rank in ascending order
    for query_id in results:
        for retrieval_type in results[query_id]:
            results[query_id][retrieval_type].sort(key=lambda x: x[1])
            results[query_id][retrieval_type] = [doc_id for doc_id, _ in results[query_id][retrieval_type]]
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trec_file', type=str, required=True)

    args = parser.parse_args()

    results = parse_trec_file(args.trec_file)
    target_ids = list(results.keys())
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    output_file_name = args.trec_file.split("/")[-1].split(".")[0]

    with open(f"eval/{output_file_name}.txt", 'w') as f, open(f"eval/{output_file_name}.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Retrieval System', 'Rank', 'ref_para_id', 'Cosine Similarity', 'Jaccard Similarity', 'Semantic Similarity', 'bleu_score', 'bert_precision', 'bert_recall', 'bert_f1'])
        
        for target_id in target_ids:
            target_paragraph = retrieve_paragraph_from_docid(target_id)
            
            f.write(f"Target ID: {target_id}\n")
            f.write(f"Target Paragraph: {target_paragraph}\n\n")

            print(f"Target ID: {target_id}")
            for retrieval_system in results[target_id].keys():
                print(f"Retrieval System: {retrieval_system}")
                for rank, ref_para_id in enumerate(results[target_id][retrieval_system], 1):
                    ref_paragraph = retrieve_paragraph_from_docid(ref_para_id)

                    cos_sim_tfidf = cosine_similarity_tfidf(target_paragraph, ref_paragraph)
                    jaccard_sim = jaccard_similarity(target_paragraph, ref_paragraph)
                    semantic_sim = semantic_similarity_bert(target_paragraph, ref_paragraph, model)
                    bleu_score = calculate_bleu_score(target_paragraph, ref_paragraph)
                    bert_precision, bert_recall, bert_f1 = calculate_bert_score(target_paragraph, ref_paragraph)
                    bert_precision, bert_recall, bert_f1 = bert_precision.item(), bert_recall.item(), bert_f1.item()

                    f.write(f"Ref ID: {ref_para_id}\n")
                    f.write(f"Rank: {rank}\n")
                    f.write(f"Retrieval System: {retrieval_system}\n")
                    f.write(f"Cosine Similarity (TF-IDF): {cos_sim_tfidf:.4f}\n")
                    f.write(f"Jaccard Similarity: {jaccard_sim:.4f}\n")
                    f.write(f"Semantic Similarity (BERT): {semantic_sim:.4f}\n")
                    f.write(f"BLEU Score: {bleu_score:.4f}\n")
                    f.write(f"BERT Score: ")
                    f.write(f"  P: {bert_precision:.4f}\n")
                    f.write(f"  R: {bert_recall:.4f}\n")
                    f.write(f"  F1: {bert_f1:.4f}\n\n")

                    print(f"Ref ID: {ref_para_id}")
                    print(f"Rank: {rank}")
                    print(f"Retrieval System: {retrieval_system}")
                    print(f"Cosine Similarity (TF-IDF): {cos_sim_tfidf:.4f}")
                    print(f"Jaccard Similarity: {jaccard_sim:.4f}")
                    print(f"Semantic Similarity (BERT): {semantic_sim:.4f}")
                    print(f"BLEU Score: {bleu_score:.4f}")
                    print(f"BERT Score: ")
                    print(f"  P: {bert_precision:.4f}")
                    print(f"  R: {bert_recall:.4f}")
                    print(f"  F1: {bert_f1:.4f}\n")

                    csv_writer.writerow([retrieval_system, rank, ref_para_id, f"{cos_sim_tfidf:.4f}", f"{jaccard_sim:.4f}", f"{semantic_sim:.4f}", f"{bleu_score:.4f}", f"{bert_precision:.4f}", f"{bert_recall:.4f}", f"{bert_f1:.4f}"])

if __name__ == '__main__':
    main()