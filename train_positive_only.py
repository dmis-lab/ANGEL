import os
import sys

import numpy as np
from tqdm import tqdm
import pickle
import argparse

import torch 
import torch.nn as nn
from transformers import TrainingArguments
from trainer import modifiedSeq2SeqTrainer
from trie import Trie
from utils import (reform_input, 
                   get_config, 
                   load_dictionary, 
                   load_trie, 
                   load_cui_label,
                   read_file_bylines,
                   convert_sets_to_lists
                   )
import copy
import json
# import wandb
import ast

def train(config):
    config.max_steps = config.max_steps // config.gradient_accumulate
    config.save_steps = config.max_steps
    
    training_args = TrainingArguments(
                    output_dir=config.model_save_path,    
                    num_train_epochs=config.num_train_epochs,                           # total number of training epochs
                    per_device_train_batch_size=config.per_device_train_batch_size,     # batch size per device during training
                    per_device_eval_batch_size=config.per_device_eval_batch_size,       # batch size for evaluation
                    warmup_steps=config.warmup_steps,                                   # number of warmup steps for learning rate scheduler
                    weight_decay=config.weight_decay,                                   # strength of weight decay
                    logging_dir=config.logging_path,                                    # directory for storing logs
                    logging_steps=config.logging_steps,
                    save_steps=config.save_steps,
                    evaluation_strategy=config.evaluation_strategy,
                    learning_rate=config.init_lr,
                    label_smoothing_factor=config.label_smoothing_factor,
                    max_grad_norm=config.max_grad_norm,
                    max_steps=config.max_steps,
                    lr_scheduler_type=config.lr_scheduler_type,
                    seed=config.seed,
                    gradient_accumulation_steps=config.gradient_accumulate, 
                    )
    if config.t5:

        from models import T5EntityPromptModel
        from transformers import T5Tokenizer, T5Config
        from datagen import prepare_trainer_dataset

        t5conf = T5Config.from_pretrained('./t5-large')
        t5conf.dropout_rate = config.dropout

        tokenizer = T5Tokenizer.from_pretrained('./t5-large')

        model = T5EntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = t5conf,
                                                    finetune = config.finetune, 
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = config.load_prompt,
                                                    soft_prompt_path = config.model_load_path,
                                                    initialize_from_vocab = config.init_from_vocab,
                                                    )
        
    else:

        from models import BartEntityPromptModel
        from transformers import BartTokenizer, BartConfig
        from datagen import prepare_trainer_dataset as prepare_trainer_dataset

        bartconf = BartConfig.from_pretrained(config.model_load_path)
        bartconf.max_position_embeddings = config.max_position_embeddings
        bartconf.attention_dropout = config.attention_dropout
        bartconf.dropout = config.dropout

        tokenizer = BartTokenizer.from_pretrained(config.model_token_path, 
                                                max_length=1024,
                                                )

        model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = bartconf,
                                                    finetune = config.finetune, 
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = config.load_prompt,
                                                    soft_prompt_path = config.model_load_path,
                                                    no_finetune_decoder = config.no_finetune_decoder,
                                                    )
        
    train_dataset, _, _ = prepare_trainer_dataset(tokenizer, 
                                                    config.dataset_path, 
                                                    prefix_mention_is = config.prefix_mention_is,
                                                    evaluate = config.evaluation,
                                                    dataset=config.dataset
                                                    )
    
    if config.unlikelihood_loss:
        print('loading trie......')
        with open(config.trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
        print('trie loaded.......')

        trainer = modifiedSeq2SeqTrainer(
                                model=model,                         # the instantiated  Transformers model to be trained
                                args=training_args,                  # training arguments, defined above
                                train_dataset=train_dataset, 
                                fairseq_loss=config.fairseq_loss,  
                                enc_num = config.prompt_tokens_enc,
                                dec_num = config.prompt_tokens_dec,
                                prefix_allowed_tokens_fn = lambda batch_id, sent: trie.get(sent.tolist()),
                                rdrop = config.rdrop,
                            )
    else:
        trainer = modifiedSeq2SeqTrainer(
                                model=model,                         # the instantiated  Transformers model to be trained
                                args=training_args,                  # training arguments, defined above
                                train_dataset=train_dataset, 
                                fairseq_loss=config.fairseq_loss,  
                                enc_num = config.prompt_tokens_enc,
                                dec_num = config.prompt_tokens_dec,
                                rdrop = config.rdrop,
                            )

    trainer.train()
    trainer.save_model(config.model_save_path)

def evalu(config):

    from fairseq_beam import SequenceGenerator, PrefixConstrainedBeamSearch, PrefixConstrainedBeamSearchWithSampling

    if config.t5:

        from models import T5EntityPromptModel
        from transformers import T5Tokenizer, T5Config
        from datagen import prepare_trainer_dataset_t5 as prepare_trainer_dataset
        
        t5conf = T5Config.from_pretrained('./t5-large')
        t5conf.dropout_rate = config.dropout

        tokenizer = T5Tokenizer.from_pretrained('./t5-large')

        model = T5EntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = t5conf,
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = True, 
                                                    soft_prompt_path = config.model_load_path
                                                    )
    
    else:

        from models import BartEntityPromptModel
        from transformers import BartTokenizer, BartConfig
        from datagen import prepare_trainer_dataset
        
        tokenizer = BartTokenizer.from_pretrained(config.model_token_path)
        bartconf = BartConfig.from_pretrained(config.model_load_path)
        bartconf.max_position_embeddings = config.max_position_embeddings
        bartconf.attention_dropout = config.attention_dropout
        bartconf.dropout = config.dropout
        bartconf.max_length = config.max_length

        model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                        config = bartconf,
                                                        n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec), 
                                                        load_prompt = True, 
                                                        soft_prompt_path=config.model_load_path
                                                        )
    
    model = model.cuda().to(model.device)
    
    train_dataset, dev_dataset, test_dataset = prepare_trainer_dataset(tokenizer, 
                                                    config.dataset_path, 
                                                    prefix_mention_is = config.prefix_mention_is,
                                                    evaluate = config.evaluation,
                                                    )

    if config.testset:
        print('eval on test set...')
        eval_dataset = test_dataset
    else:
        print('eval on develop set...')
        eval_dataset = dev_dataset
            
    cui2str, str2cui = load_dictionary(config)
    cui_labels = load_cui_label(config)
    trie = load_trie(config)
       
    # if config.rerank:
    #     print('loading retrieved names......')
    #     with open(config.retrieved_path, 'r') as f:
    #         retrieved_names = [line.split('\t')[0].split(' ') for line in f.readlines()]
    #     print('retrieved names loaded.')
    #     for i, l in tqdm(enumerate(retrieved_names)):
    #         for cui in list(l):
    #             if cui in cui2str:
    #                 continue
    #             else:
    #                 retrieved_names[i].remove(cui)

    #     print('loading tokenized names......')
    #     with open(config.dataset_path+'/tokenized.json', 'r') as f:
    #         tokenized_names = json.load(f)
    #     print('tokenized names loaded.')
        
    # if config.gold_sty:
    #     print('loading tokenized names......')
    #     with open(config.dataset_path+'/tokenized.json', 'r') as f:
    #         tokenized_names = json.load(f)
    #     print('tokenized names loaded.')

    #     print('loading sty to cui dict.....')
    #     with open(config.dataset_path+'/sty2cui.json', 'r') as f:
    #         sty2cuis = json.load(f)
    #     with open(config.dataset_path+'/sty.json', 'r') as f:
    #         cuis2sty = json.load(f)
    #     print('sty to cui dict loaded.')
        
    #     trie_dict = {}
    #     for sty in sty2cuis:
    #         names = []
    #         for cui in tqdm(sty2cuis[sty]):
    #             names += tokenized_names[cui]
    #         trie_dict[sty] = Trie(names)


    if config.wandb:
        wandb.init(project=f'{config.model_name}_finetuning')
        
    if config.beam_threshold == 0:
        print('without using beam threshold')
        beam_strategy = PrefixConstrainedBeamSearch(
            tgt_dict=None, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
        )
    else:
        beam_strategy = PrefixConstrainedBeamSearchWithSampling(
            tgt_dict=None, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            logit_thresholding=config.beam_threshold,
        )
    
    fairseq_generator = SequenceGenerator(
        models = model,
        tgt_dict = None,
        beam_size=config.num_beams,
        max_len_a=0,
        max_len_b=config.max_length,
        min_len=config.min_length,
        eos=model.config.eos_token_id,
        search_strategy=beam_strategy,

        ##### all hyperparams below are set to default
        normalize_scores=True,
        len_penalty=config.length_penalty,
        unk_penalty=0.0,
        temperature=0.7,
        match_source_len=False,
        no_repeat_ngram_size=0,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    )

    results = list()
    cui_results = list()
    results_score = list()
    given_mentions = list()
    corrects = list()
    
    input_ids = []
    decoder_input_ids = []
    attention_mask = []
    scores = {f'count_top{k+1}': 0 for k in range(10)}
    
    for i in tqdm(range(0, len(eval_dataset))):
        input_ids.append(eval_dataset[i]['input_ids'])
        attention_mask.append(eval_dataset[i]['attention_mask'])
        decoder_input_ids.append(eval_dataset[i]['decoder_input_ids_test'])

        if i % config.per_device_eval_batch_size == 0:
            
            input_ids, attention_mask = reform_input(torch.stack(input_ids), attention_mask=torch.stack(attention_mask), ending_token=model.config.eos_token_id)
            sample = {'net_input':{'input_ids':input_ids, 'attention_mask':attention_mask}}

            result_tokens, posi_scores = fairseq_generator.forward(
                sample=sample,
                prefix_mention_is = config.prefix_mention_is,
                prefix_tokens=decoder_input_ids[0].unsqueeze(0).cuda() if config.prefix_mention_is else None,
            )

            for ba, beam_sent in enumerate(result_tokens):
                result = []
                cui_result = []
                for be, sent in enumerate(beam_sent):
                    if config.prefix_mention_is:
                        result.append(tokenizer.decode(sent[len(decoder_input_ids[0]):], skip_special_tokens=True))
                    else:
                        result.append(tokenizer.decode(sent, skip_special_tokens=True))
                
                for r in result:
                    if r.strip(' ') in str2cui:
                        cui_result.append(str2cui[r.strip(' ')])
                    else:
                        cui_result.append(r)
                
                given_mention = tokenizer.decode(decoder_input_ids[0])[:-3].strip()
                given_mentions.append(given_mention)
                cui_results.append(cui_result)
                results.append(result)
                results_score.append(posi_scores)
                
                for k in range(10):
                    accumulated_results = set(cui for cuis in cui_result[:k+1] for cui in cuis)
                    if cui_labels[i].intersection(accumulated_results):
                        scores[f'count_top{k+1}'] += 1
                        if k == 0:
                            corrects.append('correct')
                    else:
                        if k == 0:
                            corrects.append('wrong')

            input_ids = []
            decoder_input_ids = []
            attention_mask = []
            
    print('=============Top1 Precision :\t',round(scores['count_top1']/(i+1)*100, 3))
    print('=============Top2 Precision :\t',round(scores['count_top2']/(i+1)*100, 3))
    print('=============Top3 Precision :\t',round(scores['count_top3']/(i+1)*100, 3))
    print('=============Top4 Precision :\t',round(scores['count_top4']/(i+1)*100, 3))
    print('=============Top5 Precision :\t',round(scores['count_top5']/(i+1)*100, 3))
    os.makedirs(config.logging_path, exist_ok=True)

    with open(f'{config.logging_path}/{config.model_name}.txt', 'a+') as f:
        f.write(config.model_load_path+'\n')
        f.write(f'=============Top1 Precision :\t{str(round(scores["count_top1"]/(i+1)*100, 3))}\n')
        f.write(f'=============Top2 Precision :\t{str(round(scores["count_top2"]/(i+1)*100, 3))}\n')
        f.write(f'=============Top3 Precision :\t{str(round(scores["count_top3"]/(i+1)*100, 3))}\n')
        f.write(f'=============Top4 Precision :\t{str(round(scores["count_top4"]/(i+1)*100, 3))}\n')
        f.write(f'=============Top5 Precision :\t{str(round(scores["count_top5"]/(i+1)*100, 3))}\n\n')
        
    zipped_list = [
    {   
        'correctness': convert_sets_to_lists(correct),
        'given_mention': convert_sets_to_lists(given_mention),
        'result': convert_sets_to_lists(result),
        'cui_label': convert_sets_to_lists(cui_label),
        'cui_result': convert_sets_to_lists(cui_result)
    }
    for correct, result, given_mention, cui_label, cui_result in zip(corrects, results, given_mentions, cui_labels, cui_results)
    ]
    
    result_score = {f'count_top{k+1}': round(scores[f'count_top{k+1}'] / (i + 1) * 100, 3) for k in range(len(scores))}
    zipped_list.insert(0, result_score)
    
    if config.testset:
        with open(os.path.join(config.logging_path, f'{config.model_load_path[15:]}_results_test_pos.json'), 'w') as f:
            json.dump(zipped_list, f, indent=2)
    else:
        with open(os.path.join(config.logging_path, f'{config.model_load_path[15:]}_results_dev_pos.json'), 'w') as f:
            json.dump(zipped_list, f, indent=2)

    return scores


if __name__ == '__main__':
    
    config = get_config()
    
    if config.evaluation:
        evalu(config)
    else:
        train(config)

