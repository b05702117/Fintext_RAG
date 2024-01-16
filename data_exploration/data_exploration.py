import json
import os
import matplotlib.pyplot as plt
from config import ROOT, RAW_DIR, FORMMATED_DIR, INDEX_DIR
import utils # utils.py
from transformers import BertTokenizer
import matplotlib.pyplot as plt

directory = "/tmp2/ybtu/FinNLP/collections/basic"
# directory = FORMMATED_DIR

# Count the total number of .jsonl files
total_files = sum(1 for f in os.listdir(directory) if f.endswith(".jsonl"))
print(f"Total number of documents: {total_files}")

# Calculate the step for each 10% of the total files
progress_step = total_files // 10

doc_count = 0
para_count = 0
total_tokens = 0
max_para_id, max_tokens = None, 0
min_para_id, min_tokens = None, float('inf')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens_per_paragraph = {}
for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(directory, filename)
        doc_count += 1
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                content = data['contents']
                tokens = tokenizer.tokenize(content)
                num_tokens = len(tokens)

                # Update
                total_tokens += num_tokens
                if num_tokens > max_tokens:
                    max_tokens = num_tokens
                    max_para_id = data['id']
                if num_tokens < min_tokens:
                    min_tokens = num_tokens
                    min_para_id = data['id']
                
                para_count += 1

                tokens_per_paragraph[data['id']] = num_tokens

        # Progress reporting every 10%
        if doc_count % progress_step == 0 or doc_count == total_files:
            percent_complete = (doc_count / total_files) * 100
            print(f"Processed {doc_count}/{total_files} files ({percent_complete:.0f}%)...")


print(f"Total number of documents: {doc_count}")
print(f"Total number of paragraphs: {para_count}")
print(f"Average number of tokens per paragraph: {total_tokens/para_count}")
print(f"para_id:{max_para_id}, Max number of tokens: {max_tokens}")
print(f"para_id:{min_para_id}, Min number of tokens: {min_tokens}")


# generate length groups
step = 100
length_groups = {}
for length in range(step, max_tokens + 1, step):
    length_groups[length] = 0
length_groups[max_tokens] = 0

for num_tokens in tokens_per_paragraph.values():
    for length in length_groups:
        if num_tokens <= length:
            length_groups[length] += 1
            break
    else:
        length_groups[length] += 1
    

# Create a dictionary with the results
results = {
    "Total number of documents": doc_count,
    "Total number of paragraphs": para_count,
    "Average number of tokens per paragraph": total_tokens / para_count if para_count else 0,
    "Paragraph ID with max tokens": max_para_id,
    "Max number of tokens": max_tokens,
    "Paragraph ID with min tokens": min_para_id,
    "Min number of tokens": min_tokens,
    "Length groups": length_groups
}

# Specify the file path for the JSON output
output_file_path = "data_exploration_results.json"

# Write the dictionary to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

with open('tokens_per_paragraph.json', 'w') as json_file:
    json.dump(tokens_per_paragraph, json_file, indent=4)

print(f"Results saved to {output_file_path}")
