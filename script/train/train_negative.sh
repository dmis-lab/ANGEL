# Basic config
#
#                             NCBI      BC5CDR    COMETA  AAP     MM-ST21pv
# Epochs                      1        1        1        1        1
# Learning Rate               1e-5     1e-5     5e-6     5e-6     5e-6
# Batch Size                  64       16       64       8        64


# bash script/train/train_negative.sh 0 ncbi 1e-5
# bash script/train/train_negative.sh 0 bc5cdr 1e-5
# bash script/train/train_negative.sh 0 cometa 5e-6
# bash script/train/train_negative.sh 0 aap 5e-6
# bash script/train/train_negative.sh 0 mm 5e-6


DEVICE_NUMBER=$1
DATASET=$2
LEARNING_RATE=$3
DATATYPE=SYN3

source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate negative_aware 

if [ $DATASET != "aap" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_negative_aware.py \
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
                                                -num_beams 10 \
                                                -dpo_topk 5 \
                                                -beta 0.1 \
                                                -num_epochs 1 \
                                                -per_device_train_batch_size 64 \
                                                -prefix_mention_is \
                                                -prefix_prompt \
                                                -evaluation \
                                                -seed 0 

else
    for split in {0..9}; do
        CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_negative_aware.py \
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
                                                    -num_beams 10 \
                                                    -dpo_topk 5 \
                                                    -beta 0.1 \
                                                    -num_epochs 1 \
                                                    -per_device_train_batch_size 64 \
                                                    -prefix_mention_is \
                                                    -prefix_prompt \
                                                    -evaluation \
                                                    -seed 0 
    done
fi
