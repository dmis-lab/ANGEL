DATASET=$1
# No model on huggingface for AskAPatient 

source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate positive_only  

CUDA_VISIBLE_DEVICES=0 python ./train_positive_only.py \
        -dataset_path ./benchmarks/$DATASET \
        -model_load_path dmis-lab/ANGEL_$DATASET \
        -model_token_path facebook/bart-large \
        -trie_path ./benchmarks/$DATASET/trie.pkl\
        -dict_path ./benchmarks/"$DATASET"/target_kb.json \
        -model_name $DATASET \
        -per_device_eval_batch_size 1 \
        -num_beams 10 \
        -prefix_prompt \
        -prefix_mention_is \
        -evaluation \
        -testset
