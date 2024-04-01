from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.readers import InputExample
import torch
import json
import argparse
import os
import random
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2-tfidf")
parser.add_argument("--output_path", type=str, default="model/")
# TODO: 要加個output file name 跟get index name 對接
# parser.add_argument("--checkpoint_path", type=str, default="model/")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--device", type=str, default="0")


args = parser.parse_args()

def load_input_examples(positive_data_path="positive_pairs.jsonl", negative_data_path="negative_pairs.jsonl", validation_split=0.1):
    train_examples = []

    with open(positive_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            train_examples.append(InputExample(texts=[data['target'], data['reference']], label=float(1)))

    with open(negative_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            train_examples.append(InputExample(texts=[data['target'], data['reference']], label=float(0)))
    
    random.shuffle(train_examples)

    # Split train and validation
    total_examples = len(train_examples)
    validation_size = int(total_examples * validation_split)
    validation_examples = train_examples[:validation_size]
    train_examples = train_examples[validation_size:]

    print("Total examples: ", total_examples)
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")

    return train_examples, validation_examples

def main():
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S', filename='training_log.log', filemode='w')

    model = SentenceTransformer("all-mpnet-base-v2")

    train_examples, validation_examples = load_input_examples()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(validation_examples)
    evaluation_steps = len(train_examples) // args.batch_size

    logging.info("Starting the training process")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=args.epochs,
        warmup_steps=100,
        output_path=os.path.join(args.output_path, args.model_name), 
        evaluator=evaluator,
        evaluation_steps=evaluation_steps
    )

    model.save(os.path.join(args.output_path, args.model_name))
    # model.save(f"{args.output_path}/{args.model_name}")

if __name__ == "__main__":
    main()