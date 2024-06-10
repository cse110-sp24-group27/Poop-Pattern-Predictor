import json
import csv
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Paths to your files
jsonl_file_path = 'data/obqa/statement/test-fact.statement.jsonl'
csv_file_path = 'saved_models/test_preds_obqa.csv'

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    last_hidden_state = outputs[0]
    return last_hidden_state.mean(dim=1)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_csv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            question_id = row[0]
            predicted_answer = row[1]
            data[question_id] = predicted_answer
    return data

def get_choice_text(choices, label):
    for choice in choices:
        if choice['label'] == label:
            return choice['text']
    return ""

def calculate_sas(ground_truth, predicted):
    gt_embedding = embed_text(ground_truth)
    pred_embedding = embed_text(predicted)
    similarity = cosine_similarity(gt_embedding.detach().numpy(), pred_embedding.detach().numpy())[0][0]
    return similarity

def plot_histogram(sas_scores):
    plt.hist(sas_scores, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('Semantic Answer Similarity (SAS)')
    plt.ylabel('Frequency')
    plt.title('Distribution of SAS Scores')
    plt.show()

def main(jsonl_file_path, csv_file_path):
    # Load data
    questions = load_jsonl(jsonl_file_path)
    predictions = load_csv(csv_file_path)

    # Calculate SAS for each question
    sas_scores = []
    filtered_entries = []
    count = 0
    q_len = len(questions)
    for question in questions:
        question_id = question['id']
        ground_truth_label = question['answerKey']
        ground_truth_answer = get_choice_text(question['question']['choices'], ground_truth_label)
        predicted_label = predictions.get(question_id, "")
        predicted_answer = get_choice_text(question['question']['choices'], predicted_label)

        if predicted_answer:
            sas = calculate_sas(ground_truth_answer, predicted_answer)
            sas_scores.append(sas)
            print(f'Question ID: {question_id} - SAS: {sas:.4f}')
            if predicted_answer != ground_truth_answer:
                count += 1
                filtered_entries.append({
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "sas_score": sas
                })

    # Calculate average SAS score
    filtered_sas_scores = [score for score in sas_scores if score < 1.000]
    average_sas = sum(sas_scores) / len(sas_scores) if sas_scores else 0
    average_sasWrong = sum(filtered_sas_scores) / len(filtered_sas_scores) if filtered_sas_scores else 0
    print(f'Average SAS Score: {average_sas:.4f}')
    print(f'Average of Wrong SAS Score: {average_sasWrong:.4f}')
    print(count)
    print(q_len)
    
    # plot_histogram(sas_scores)

    # Write filtered entries to a txt file
    output_txt_file_path = 'filtered_sas_scores_obqa.txt'
    with open(output_txt_file_path, 'w', encoding='utf-8') as f:
        f.write(str(count) + "\n")
        f.write(str(q_len) + "\n")
        for entry in filtered_entries:
            f.write(f"Question ID: {entry['question']['id']}\n")
            f.write(f"Question: {entry['question']['question']['stem']}\n")
            f.write(f"Ground Truth Answer: {get_choice_text(entry['question']['question']['choices'], entry['question']['answerKey'])}\n")
            f.write(f"Predicted Answer: {entry['predicted_answer']}\n")
            f.write(f"SAS Score: {entry['sas_score']:.4f}\n")
            f.write("Choices:\n")
            for choice in entry['question']['question']['choices']:
                f.write(f" - {choice['label']}: {choice['text']}\n")
            f.write("\n")

if __name__ == "__main__":
    main(jsonl_file_path, csv_file_path)
