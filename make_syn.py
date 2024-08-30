import os
import json
import random
import numpy as np
import joblib
from tqdm import tqdm
import argparse

# Load the UMLS TF-IDF vectorizer
umls_vectorizer = joblib.load('./umls_tfidf_vectorizer.joblib')


def read_file_by_lines(file: str) -> list:
    """Read a file and return its lines without newline characters."""
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_datas(src_path: str, trg_path: str, label_path: str):
    """Load data from source, target, and label files."""
    with open(label_path, 'r') as f:
        cui_labels = [set(cui.strip().replace('+', '|').split('|')) for cui in f.readlines()]
    mentions = [json.loads(x)[0][:-3] for x in read_file_by_lines(trg_path)]
    sources = [json.loads(x) for x in read_file_by_lines(src_path)]
    return cui_labels, mentions, sources


def calculate_similarity_tfidf_top_k(a_features, b_features, k: int) -> list:
    """Calculate the top-k most similar indices between two feature sets using TF-IDF."""
    sim = np.array(b_features.dot(a_features.T).todense())[0]
    top_k_indices = np.argsort(sim)[-k:][::-1]
    return top_k_indices


def find_values_with_keyword(input_dict: dict, CUI: str) -> list:
    """Find values in a dictionary with a given keyword."""
    contained_keys = [key for key in input_dict if CUI in key]
    values = [value for key in contained_keys for value in input_dict[key]]
    return values


def main(args):
    """Main function to process data and write to output files."""
    # Load the target knowledge base
    with open(f'./benchmarks/{args.dataset}/target_kb.json') as f:
        cui2str = json.load(f)
    
    src_path=f'./benchmarks/{args.dataset}/train.source'
    trg_path=f'./benchmarks/{args.dataset}/train.target'
    label_path=f'./benchmarks/{args.dataset}/trainlabel.txt'
    
    # Load training data
    train_label, train_mention, train_src = load_datas(
                                                        src_path=src_path, 
                                                        trg_path=trg_path,
                                                        label_path=label_path
                                                        )

    output_dir_all = f'./benchmarks/{args.dataset}_SYN{args.num_k}'
    os.makedirs(output_dir_all, exist_ok=True)

    # Open files for appending in both directories
    with open(os.path.join(output_dir_all, "train.source"), 'a') as f1_all, \
         open(os.path.join(output_dir_all, "train.target"), 'a') as f2_all, \
         open(os.path.join(output_dir_all, "trainlabel.txt"), 'a') as f3_all:

        # Iterate over the training data
        for i in tqdm(range(len(train_label))):
            input_data = train_src[i]
            mention = train_mention[i]
            cuis = list(train_label[i])

            # Generate candidate list
            candidate_list = [name for cui in cuis for name in find_values_with_keyword(cui2str, cui)]
            mention_vector = umls_vectorizer.transform([mention])
            positive_indices = calculate_similarity_tfidf_top_k(umls_vectorizer.transform(candidate_list), mention_vector, args.num_k)
            positives = [candidate_list[idx] for idx in positive_indices]

            for k in range(min(args.num_k, len(positives))):
                f1_all.write(json.dumps(input_data) + '\n')
                f2_all.write(json.dumps([f'{mention} is', f'{positives[k]}']) + '\n')
                f3_all.write(json.dumps(cuis) + '\n')
                
        print(f"Finished making {args.dataset}")
        

def main_aap(args):
    """Main function to process data and write to output files."""
    # Load the target knowledge base
    with open(f'./benchmarks/{args.dataset}/target_kb.json') as f:
        cui2str = json.load(f)
    
    for num in range(10):
        src_path=f'./benchmarks/{args.dataset}/fold{num}/train.source'
        trg_path=f'./benchmarks/{args.dataset}/fold{num}/train.target'
        label_path=f'./benchmarks/{args.dataset}/fold{num}/trainlabel.txt'

        # Load training data
        train_label, train_mention, train_src = load_datas(
                                                            src_path=src_path, 
                                                            trg_path=trg_path,
                                                            label_path=label_path
                                                            )

        output_dir_all = f'./benchmarks/{args.dataset}/fold{num}_SYN{args.num_k}'
        os.makedirs(output_dir_all, exist_ok=True)

        # Open files for appending in both directories
        with open(os.path.join(output_dir_all, "train.source"), 'a') as f1_all, \
            open(os.path.join(output_dir_all, "train.target"), 'a') as f2_all, \
            open(os.path.join(output_dir_all, "trainlabel.txt"), 'a') as f3_all:

            # Iterate over the training data
            for i in tqdm(range(len(train_label))):
                input_data = train_src[i]
                mention = train_mention[i]
                cuis = list(train_label[i])

                # Generate candidate list
                candidate_list = [name for cui in cuis for name in find_values_with_keyword(cui2str, cui)]
                mention_vector = umls_vectorizer.transform([mention])
                positive_indices = calculate_similarity_tfidf_top_k(umls_vectorizer.transform(candidate_list), mention_vector, args.num_k)
                positives = [candidate_list[idx] for idx in positive_indices]

                for k in range(min(args.num_k, len(positives))):
                    f1_all.write(json.dumps(input_data) + '\n')
                    f2_all.write(json.dumps([f'{mention} is', f'{positives[k]}']) + '\n')
                    f3_all.write(json.dumps(cuis) + '\n')
                    
            print(f"Finished making {args.dataset} fold{num}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and generate synthetic training data.")
    parser.add_argument("-dataset", type=str, default='ncbi', help="Dataset name")
    parser.add_argument("-num_k", type=int, default=3, help="Number of top candidates to consider")
    args = parser.parse_args()
    
    if args.dataset == "aap":
        main_aap(args)
    else:
        main(args)
