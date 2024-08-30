from transformers import BartConfig
from trl import DPOTrainer, DPOConfig
from models import BartEntityPromptModel
from fairseq_beam import SequenceGenerator, PrefixConstrainedBeamSearch
from torch.utils.data import Subset
import torch
import copy
import random
from utils import *
from datagen import prepare_trainer_dataset
import numpy as np
import os
import shutil
import joblib

umls_vectorizer = joblib.load('./umls_tfidf_vectorizer.joblib')

def calculate_similarity_tfidf_top_3(a_features, b_features) -> list:
    sim = np.array(b_features.dot(a_features.T).todense())[0]
    top_3_indices = np.argsort(sim)[-3:][::-1]
    return top_3_indices

def evaluate(model, tokenizer, eval_dataset, trie, str2cui, cui2str, config, set_name):
    # Load BART configuration and set necessary parameters
    bartconf = BartConfig.from_pretrained(config.model_load_path)
    bartconf.max_position_embeddings = config.max_position_embeddings
    bartconf.attention_dropout = config.attention_dropout
    bartconf.dropout = config.dropout
    bartconf.max_length = config.max_length
    
    # Load the evaluation model with prompts and configuration
    eval_model = BartEntityPromptModel.from_pretrained(
        config.model_load_path, 
        config=bartconf,
        n_tokens=(config.prompt_tokens_enc, config.prompt_tokens_dec), 
        load_prompt=True, 
        soft_prompt_path=config.model_load_path,
    )
    
    # Load the state of the original model into the evaluation model
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()  # Set model to evaluation mode
    eval_model.cuda().to(eval_model.device)  # Move model to GPU if available
    
    # Initialize the beam search strategy with prefix constraints from the Trie
    beam_strategy = PrefixConstrainedBeamSearch(
        tgt_dict=None, 
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
    )
    
    # Configure the sequence generator for the evaluation
    fairseq_generator = SequenceGenerator(
        models=eval_model,
        tgt_dict=None,
        beam_size=config.num_beams,
        max_len_a=0,
        max_len_b=config.max_length,
        min_len=config.min_length,
        eos=eval_model.config.eos_token_id,
        search_strategy=beam_strategy,

        ##### All hyperparameters below are set to their default values
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

    # Initialize lists to store results and scores
    results = []
    cui_results = []
    results_score = []
    given_mentions = []
    corrects = []
    
    # Initialize input and score tracking variables
    input_ids = []
    decoder_input_ids = []
    attention_mask = []
    scores = {f'count_top{k+1}': 0 for k in range(config.num_beams)}
    output_for_dpo = defaultdict(list)
    
    # Iterate through the evaluation dataset
    for i in tqdm(range(0, len(eval_dataset))):
        if eval_dataset[i][1]:
            # Collect input IDs, attention masks, and decoder input IDs
            input_ids.append(eval_dataset[i][0]['input_ids'])
            attention_mask.append(eval_dataset[i][0]['attention_mask'])
            decoder_input_ids.append(eval_dataset[i][0]['decoder_input_ids_test'])
            
            # Process in batches
            if i % config.per_device_eval_batch_size == 0:
                # Prepare input for the model
                input_ids, attention_mask = reform_input(
                    torch.stack(input_ids), 
                    attention_mask=torch.stack(attention_mask), 
                    ending_token=model.config.eos_token_id
                )
                sample = {'net_input': {'input_ids': input_ids, 'attention_mask': attention_mask}}
                                    
                # Generate results using the model
                result_tokens, posi_scores = fairseq_generator.forward(
                    sample=sample,
                    prefix_mention_is=config.prefix_mention_is,
                    prefix_tokens=decoder_input_ids[0].unsqueeze(0).cuda() if config.prefix_mention_is else None,
                )
                                
                # Decode the results and map to CUI labels
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
                    
                    # Store the given mention and results
                    given_mention = tokenizer.decode(decoder_input_ids[0])[:-3].strip()
                    given_mentions.append(given_mention)
                    cui_results.append(cui_result)
                    results.append(result)
                    results_score.append(posi_scores)
                    accumulated_results = set()
                    answer_with_correctness = []
                    no_answer = True

                    # 1. Collect correct and incorrect answers
                    for k, cui in enumerate(cui_result[:config.dpo_topk]):
                        if eval_dataset[i][1].intersection(cui):
                            answer_with_correctness.append(("correct", result[k]))
                            no_answer = False
                        else:
                            answer_with_correctness.append(("wrong", result[k]))

                    # Handle case when no correct answer is found
                    if no_answer:
                        correct_candidates = cui2str[list(eval_dataset[i][1])[0]]
                        mention_vector = umls_vectorizer.transform([given_mention])
                        positive_indices = calculate_similarity_tfidf_top_3(umls_vectorizer.transform(correct_candidates), mention_vector)
                        candidates = [correct_candidates[idx] for idx in positive_indices]

                    # 2. Update accumulated results and scores
                    for k in range(config.num_beams):
                        accumulated_results.update(cui for cuis in cui_result[:k+1] for cui in cuis)
                        if eval_dataset[i][1].intersection(accumulated_results):
                            scores[f'count_top{k+1}'] += 1
                            if k == 0:
                                corrects.append('correct')
                        else:
                            if k == 0:
                                corrects.append('wrong')

                    # Prepare prompt text
                    prompt_text = tokenizer.decode(input_ids[0]).strip()[4:-4]

                    # Store DPO output
                    for k, word in enumerate(answer_with_correctness):
                        if word[0] == "correct":
                            if k == 0:
                                if "wrong" in [pair[0] for pair in answer_with_correctness]:
                                    output_for_dpo['prompt'].append(prompt_text)
                                    output_for_dpo['chosen'].append(f'{given_mention} is {word[1]}')
                                    output_for_dpo['rejected'].append(f'{given_mention} is {[pair[1] for pair in answer_with_correctness if pair[0]=="wrong"][0]}')
                                    continue
                            for j in range(k):
                                if answer_with_correctness[j][0] == "wrong":
                                    output_for_dpo['prompt'].append(prompt_text)
                                    output_for_dpo['chosen'].append(f'{given_mention} is {word[1]}')
                                    output_for_dpo['rejected'].append(f'{given_mention} is {answer_with_correctness[j][1].strip()}')
                    
            # Reset input IDs, attention masks, and decoder input IDs
            input_ids = []
            decoder_input_ids = []
            attention_mask = []
    
    # Print final precision scores for the dataset
    print(f'{set_name} set FINAL SCORE!\t')
    print('=============Top1 Precision :\t', round(scores['count_top1'] / (i+1) * 100, 3))
    print('=============Top2 Precision :\t', round(scores['count_top2'] / (i+1) * 100, 3))
    print('=============Top3 Precision :\t', round(scores['count_top3'] / (i+1) * 100, 3))
    print('=============Top4 Precision :\t', round(scores['count_top4'] / (i+1) * 100, 3))
    print('=============Top5 Precision :\t', round(scores['count_top5'] / (i+1) * 100, 3))
    
    # Store the precision scores in a dictionary
    result_score = {f'count_top{k+1}': round(scores[f'count_top{k+1}'] / (i + 1) * 100, 3) for k in range(len(scores))}
    
    # If evaluating on the test set, save detailed results to a JSON file
    if set_name == 'Test':
        def convert_sets_to_lists(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_sets_to_lists(value) for key, value in obj.items()}
            else:
                return obj
            
        # Zip together results and store them in a list
        zipped_list = [
        {   
            'correctness': convert_sets_to_lists(correct),
            'given_mention': convert_sets_to_lists(given_mention),
            'result': convert_sets_to_lists(result),
            'cui_label': convert_sets_to_lists(eval_data[1]),
            'cui_result': convert_sets_to_lists(cui_result)
        }
        for correct, result, given_mention, eval_data, cui_result in zip(corrects, results, given_mentions, eval_dataset, cui_results)
        ]
        
        # Insert the result scores at the beginning of the list
        zipped_list.insert(0, result_score)

        # Save the results to a JSON file
        with open(os.path.join(config.model_load_path, 'results_test_neg.json'), 'w') as f:
            json.dump(zipped_list, f, indent=2)
    
    return Dataset.from_dict(dict(output_for_dpo)), result_score



def train(model, tokenizer, train_dataset, config):

    model_ref = copy.deepcopy(model)
    model_ref.to(device)
    
    training_args = DPOConfig(
        report_to='none',
        output_dir=config.model_save_path,
        beta=config.beta,
        weight_decay=0.01,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.init_lr,
        num_train_epochs=1,
        logging_steps=config.logging_steps,
        lr_scheduler_type='polynomial',
        run_name=config.model_name,
        is_encoder_decoder=True,
        max_length=256,
        max_prompt_length=256,
        max_target_length=256,
        remove_unused_columns=False,
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    
    dpo_trainer.train()
    
    return dpo_trainer.model


def main():
    
    # Get configuration settings
    config = get_config()
    random.seed(config.seed)
    
    # Load model and tokenizer for DPO (Direct Preference Optimization)
    model, tokenizer = load_model(config, dpo=True)
    model.to(device)
     
    # Load datasets for training, validation (dev), and testing
    train_dataset, dev_dataset, test_dataset = prepare_trainer_dataset(tokenizer, 
                                                                       config.dataset_path, 
                                                                       prefix_mention_is=config.prefix_mention_is,
                                                                       evaluate=True)

    # Load labels for the datasets
    train_cui_labels = load_label_ft(config, 'train')
    dev_cui_labels = load_label_ft(config, 'dev')
    test_cui_labels = load_label_ft(config, 'test')

    # Pair datasets with their corresponding labels
    train_pair = list(zip(train_dataset, train_cui_labels))
    random.shuffle(train_pair)  # Shuffle training pairs
    dev_pair = list(zip(dev_dataset, dev_cui_labels))
    test_pair = list(zip(test_dataset, test_cui_labels))

    # Load CUI to string dictionary and Trie data structure
    cui2str, str2cui = load_dictionary(config)
    trie = load_trie(config)
    
    # Split the training pairs into smaller chunks for cross-validation
    n_splits = 10
    train_chunks = [train_pair[i::n_splits] for i in range(n_splits)]
    
    # Initialize variables for early stopping
    previous_score = 0
    patience = 3
    num_no_improvement = 0
        
    # Training loop
    for _ in range(config.num_epochs):
        for num, splitted_dataset in enumerate(train_chunks):
            print(f"Start finetuning with {num+1} / {len(train_chunks)} subset!")
            
            # Evaluate the model on the validation (dev) set to get initial scores
            _, dev_scores = evaluate(model, tokenizer, dev_pair, trie, str2cui, cui2str, config, 'Develop')
            
            # Generate a DPO training dataset from the current chunk of training data
            dpo_dataset_for_train, _ = evaluate(model, tokenizer, splitted_dataset, trie, str2cui, cui2str, config, 'Train')
            
            # Train the model on the DPO training dataset
            model = train(model, tokenizer, dpo_dataset_for_train, config)

            # Define the path to save the trained model
            model_save_path = f'{config.model_save_path}/{config.init_lr}_{config.beta}_{config.per_device_train_batch_size}'
            
            # Early stopping mechanism
            # If the top-1 score on the validation set improves, save the model
            if dev_scores['count_top1'] > previous_score:
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                    
                # Remove any existing saved models in the directory
                for filename in os.listdir(model_save_path):
                    file_path = os.path.join(model_save_path, filename)
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                
                # Save the current model, tokenizer, and configuration
                model.save_pretrained(f'{model_save_path}/{num}_epoch', safe_serialization=False)
                tokenizer.save_pretrained(f'{model_save_path}/{num}_epoch')
                model.config.save_pretrained(f'{model_save_path}/{num}_epoch')
                
                # Reset the no improvement counter
                num_no_improvement = 0
            else:
                # If there is no improvement, increment the counter
                num_no_improvement += 1
                print(f"No improvement in development score for {num_no_improvement} consecutive rounds.")
                
                # If the number of rounds without improvement exceeds patience, stop training
                if num_no_improvement >= patience:
                    print(f"Early stopping after {num_no_improvement} rounds with no improvement.")
                    break
                
            # Update the previous score with the current top-1 score
            previous_score = dev_scores['count_top1'] 
                
        # If early stopping was triggered, exit the loop
        if num_no_improvement >= patience:
            break
        
    # After training is complete, evaluate the model on the test set
    print("Training Completed.")
    config.model_load_path = [f.path for f in os.scandir(model_save_path) if f.is_dir()][0]
    model, tokenizer = load_model(config, dpo=False)
    _, test_scores = evaluate(model, tokenizer, test_pair, trie, str2cui, cui2str, config, 'Test')
    
    
    print(test_scores)
    
if __name__ == '__main__':
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run the main function
    main()
