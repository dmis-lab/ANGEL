# Basic config
#
#                             NCBI      BC5CDR    COMETA  AAP     MM-ST21pv
# Epochs                      1        1        1        1        1
# Learning Rate               1e-5     1e-6     5e-6     5e-6     5e-6
# Batch Size                  16       16       32       8        16


# bash script/train/train_negative.sh ncbi 1e-5 16
# bash script/train/train_negative.sh bc5cdr 1e-6 16
# bash script/train/train_negative.sh cometa 5e-6 32
# bash script/train/train_negative.sh aap 5e-6 8
# bash script/train/train_negative.sh mm 5e- 16


DATASET=$1
LEARNING_RATE=$2
BATCH_SIZE=$3
DATATYPE=SYN3

source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate negative_aware 

if [ $DATASET != "aap" ]; then
    python ./train_negative_aware.py \
            -dataset_path ./benchmarks/$DATASET \
            -model_name Negative_aware-"$DATASET" \
            -model_load_path ./model/positive_only/"$DATASET"_"$DATATYPE" \
            -model_save_path ./model/negative_aware/"$DATASET"_"$DATATYPE" \
            -trie_path ./benchmarks/"$DATASET"/trie.pkl \
            -dict_path ./benchmarks/"$DATASET"/target_kb.json \
            -model_token_path facebook/bart-large \
            -logging_path ./logs/negative_aware/$DATASET \
            -init_lr $LEARNING_RATE \
            -logging_steps 500 \
            -num_beams 5 \
            -dpo_topk 5 \
            -beta 0.1 \
            -num_epochs 1 \
            -per_device_train_batch_size $BATCH_SIZE \
            -prefix_mention_is \
            -prefix_prompt \
            -evaluation \
            -seed 0 

else
    for split in {0..9}; do
        python ./train_negative_aware.py \
                -dataset_path ./benchmarks/$DATASET/fold$split \
                -model_name Negative_aware-"$DATASET" \
                -model_load_path ./model/positive_only/"$DATASET"_"$DATATYPE"/fold$split \
                -model_save_path ./model/negative_aware/"$DATASET"_"$DATATYPE" \
                -trie_path ./benchmarks/"$DATASET"/trie.pkl \
                -dict_path ./benchmarks/"$DATASET"/target_kb.json \
                -model_token_path facebook/bart-large \
                -logging_path ./logs/negative_aware/$DATASET \
                -init_lr $LEARNING_RATE \
                -logging_steps 500 \
                -num_beams 5 \
                -dpo_topk 5 \
                -beta 0.1 \
                -num_epochs 1 \
                -per_device_train_batch_size $BATCH_SIZE \
                -prefix_mention_is \
                -prefix_prompt \
                -evaluation \
                -seed 0 
    done
fi
