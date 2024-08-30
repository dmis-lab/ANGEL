import torch
import pickle
import argparse
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from models import BartEntityPromptModel
from collections import defaultdict
import json
from tqdm import tqdm
import random
from datasets import Dataset
from trie import Trie

def load_dictionary(config):
    
    print('loading dictionary....')
    dict_path = config.dict_path
    if 'json' in dict_path:
        with open(dict_path, 'r') as f:
            cui2str = json.load(f)
    else:
        with open(dict_path, 'rb') as f:
            cui2str = pickle.load(f)

    str2cui = {}
    for cui in cui2str:
        if isinstance(cui2str[cui], list):
            for name in cui2str[cui]:
                if name in str2cui:
                    str2cui[name].append(cui)
                else:
                    str2cui[name] = [cui]
        else:
            name = cui2str[cui]
            if name in str2cui:
                str2cui[name].append(cui)
                print('duplicated vocabulary')
            else:
                str2cui[name] = [cui]
    print('dictionary loaded......')
    
    return cui2str, str2cui
    
def load_trie(config):
    
    print('loading trie......')
    with open(config.trie_path, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    return trie

def load_cui_label(config):
    
    print('loading cui labels......')
    with open(config.dataset_path+f'/{["test" if config.testset else "dev"][0]}label.txt', 'r') as f:
        cui_labels = [set(cui.strip('\n').replace('+', '|').split('|')) for cui in f.readlines()]

    return cui_labels

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    else:
        return obj

def load_label_ft(config, set_type):
    
    print(f'loading {set_type} label cuis......')
    with open(config.dataset_path+f'/{set_type}label.txt', 'r') as f:
        cui_labels = [set(cui.strip('\n').replace('+', '|').split('|')) for cui in f.readlines()]
    print(f'{set_type} label cuis loaded')

    return cui_labels

def reform_input(inputs, attention_mask = None, ending_token = 2):
    
    ## input a tensor of size BSZ x Length
    max_idx = torch.max(torch.where(inputs==ending_token)[1])
    inputs = inputs[:, :max_idx+1]
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_idx+1]

    return inputs, attention_mask

def read_file_bylines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    output = []
    for item in lines[:]:
        output.append(ast.literal_eval(item.strip('\n')))
    return output

def load_model(config, dpo=False):

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', max_length=1024)
    bartconf = BartConfig.from_pretrained(config.model_load_path)
    
    if dpo:
        model = BartForConditionalGeneration.from_pretrained(config.model_load_path, config = bartconf)
    else:
        model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                        config = bartconf,
                                                        n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec), 
                                                        load_prompt = True, 
                                                        soft_prompt_path=config.model_load_path,
                                                        use_safetensors=True
                                                        )
    return model, tokenizer

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    else:
        return obj
            
def get_config():
    parser = argparse.ArgumentParser(description='Training configuration')

    # Dataset paths
    parser.add_argument("-dataset_path", type=str, help="Path of the medmentions dataset")
    parser.add_argument("-dpo_dataset_path", type=str, help="Path of the DPO dataset")
    
    # Model configuration
    parser.add_argument("-model_name", type=str, default='ncbi', help="Name of the model")
    parser.add_argument("-dataset", type=str, default='', help="Choosing dataset for fine-tuning")
    parser.add_argument("-model_save_path", type=str, default='./model_saved', help="Path to save the trained model")
    parser.add_argument("-model_load_path", type=str, default='facebook/bart-large', help="Path to load the pretrained model")
    parser.add_argument("-model_token_path", type=str, default='facebook/bart-large', help="Path for tokenizer")
    parser.add_argument("-trie_path", type=str, default='./trie.pkl', help="Path of the Trie")
    parser.add_argument("-dict_path", type=str, default='./benchmark/ncbi_EL/target_kb.json', help="Path of the CUI to string dictionary")
    parser.add_argument("-retrieved_path", type=str, default='./trie.pkl', help="Path of the retrieved CUI to string dictionary")
    
    # Logging and saving
    parser.add_argument("-logging_path", type=str, default='./logs', help="Path for saving logs")
    parser.add_argument('-logging_steps', type=int, default=1000, help='Save logs per logging step')
    parser.add_argument('-save_steps', type=int, default=20000, help='Save checkpoints per save steps')
    parser.add_argument('-eval_steps', type=int, default=500, help='evaluation steps')

    # Training parameters
    parser.add_argument('-num_train_epochs', type=int, default=8, help="Number of training epochs")
    parser.add_argument('-per_device_train_batch_size', type=int, default=4, help="Training batch size")
    parser.add_argument('-per_device_eval_batch_size', type=int, default=1, help="Evaluation batch size")
    parser.add_argument('-warmup_steps', type=int, default=500, help="Number of warmup steps")
    parser.add_argument('-max_grad_norm', type=float, default=0.1, help="Gradient clipping value")
    parser.add_argument('-max_steps', type=int, default=20000, help="Max training steps, overrides num_train_epochs")
    parser.add_argument('-gradient_accumulate', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('-weight_decay', type=float, default=0.01, help="Weight decay of the optimizer")
    parser.add_argument('-init_lr', type=float, default=5e-5, help="Initial learning rate for AdamW")
    parser.add_argument('-lr_scheduler_type', type=str, default='polynomial', help="Learning rate scheduler type")
    parser.add_argument('-evaluation_strategy', type=str, default='no', help="Evaluation strategy")

    # Beam search parameters
    parser.add_argument('-num_beams', type=int, default=5, help="Number of beams for beam search")
    parser.add_argument('-length_penalty', type=float, default=1, help="Length penalty of beam search")
    parser.add_argument('-beam_threshold', type=float, default=0, help="Logit threshold for beam search")
    parser.add_argument('-max_length', type=int, default=1024, help="Max length for generation")
    parser.add_argument('-min_length', type=int, default=1, help="Min length for generation")

    # Dropout and regularization
    parser.add_argument('-attention_dropout', type=float, default=0.1, help="Attention dropout")
    parser.add_argument('-dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('-rdrop', type=float, default=0.0, help="R-drop regularization")
    parser.add_argument('-label_smoothing_factor', type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument('-unlikelihood_loss', action='store_true', help="Whether to use unlikelihood loss")
    parser.add_argument('-unlikelihood_weight', type=float, default=0.1, help="Weight for unlikelihood loss")

    # Model architecture
    parser.add_argument('-max_position_embeddings', type=int, default=1024, help="Max position embedding length")
    parser.add_argument('-prompt_tokens_enc', type=int, default=0, help="Number of soft prompt tokens in encoder")
    parser.add_argument('-prompt_tokens_dec', type=int, default=0, help="Number of soft prompt tokens in decoder")
    
    # Special training modes and options
    parser.add_argument('-make_all_pair', action='store_true', help="Create all possible pairs for training")
    parser.add_argument('-finetune', action='store_true', help="Finetune the BART model")
    parser.add_argument('-t5', action='store_true', help="Use T5 pretrained model")
    parser.add_argument('-fairseq_loss', action='store_true', help="Use label smoothed loss in fairseq")
    parser.add_argument('-evaluation', action='store_true', help="Set to evaluation mode")
    parser.add_argument('-testset', action='store_true', help="Evaluate with test set or dev set")
    parser.add_argument('-load_prompt', action='store_true', help="Load prompt during training")
    parser.add_argument('-sample_train', action='store_true', help="Use training target sampled by TF-IDF similarity")
    parser.add_argument('-prefix_prompt', action='store_true', help="Use prefix prompt tokens")
    parser.add_argument('-init_from_vocab', action='store_true', help="Initialize prompt from mean of token embeddings")
    parser.add_argument('-rerank', action='store_true', help="Rerank the retrieved names")
    parser.add_argument('-no_finetune_decoder', action='store_true', help="Only finetune encoder")
    parser.add_argument('-syn_pretrain', action='store_true', help="Pretrain on synthetic data")
    parser.add_argument('-new_dpo_method', action='store_true', help="Use new DPO method for training")
    parser.add_argument('-gold_sty', action='store_true', help="Use gold stylistic data")
    parser.add_argument('-prefix_mention_is', action='store_true', help="Use prefix 'mention is'")

    # DPO-specific configurations
    parser.add_argument("-num_epochs", type=int, default=1, help="Number of training epochs for DPO")
    parser.add_argument("-beta", type=float, default=0.1, help="Beta value for DPO")
    parser.add_argument('-dpo_topk', type=int, default=10, help="Number of reranking candidates")
    
    # Debugging and special modes
    parser.add_argument('-wandb', action='store_true', help="Use wandb for logging")
    parser.add_argument("-sweep", action='store_true', help="Enable sweep debugging mode for DPO")
    parser.add_argument("-hard_negative", action='store_true', help="Enable hard negative mode for DPO")
    parser.add_argument("-debug", action='store_true', help="Enable debugging mode")
    
    # Distributed training
    parser.add_argument("-local_rank", type=int, default=0, help="Local rank for distributed training")
    
    # Random seed for reproducibility
    parser.add_argument("-seed", type=int, default=0, help="Seed for random number generators to ensure reproducibility")


    config = parser.parse_args()
    return config