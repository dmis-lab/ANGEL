import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import json
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer
from trie import Trie


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def main(args):
    with open(f'./{args.base_dir}/{args.dataset}/target_kb.json', 'r') as f:
        cui2str = json.load(f)

    entities = []
    for cui in cui2str:
        entities += cui2str[cui]
    trie = Trie([16]+list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in tqdm(entities)).trie_dict
    with open(f'./{args.base_dir}/{args.dataset}/trie.pkl', 'wb') as w_f:
        pickle.dump(trie, w_f)
    print("finish running!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_dir", type=str, default='./benchmarks', help="Dataset name")
    parser.add_argument("-dataset", type=str, default='ncbi', help="Dataset name")
    args = parser.parse_args()
    
    main(args)