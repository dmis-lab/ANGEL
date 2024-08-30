# Basic config
#
#                             NCBI      BC5CDR    COMETA  AAP     MM-ST21pv
# Training Steps              20000     30000     40000   30000   40000
# Learning Rate               3e-7      5e-6      5e-6    5e-6    2e-5
# Weight Decay                0.01      0.01      0.01    0.01    0.01
# Batch Size                  16        16        16      16      16
# Warmup Steps                0         500       500     0       1,000


# bash script/train/train_positive.sh 0 ncbi 3e-7 20000
# bash script/train/train_positive.sh 0 bc5cdr 5e-6 30000
# bash script/train/train_positive.sh 0 cometa 5e-6 40000
# bash script/train/train_positive.sh 0 aap 5e-6 30000
# bash script/train/train_positive.sh 0 mm 2e-5 40000


DEVICE_NUMBER=$1
DATASET=$2
LEARNING_RATE=$3
STEPS=$4
BATCH_SIZE= 64
MODEL_PATH=chanwhistle/ANGEL_Pretrained
DATATYPE=SYN3                                   # If you want to don't use top-k synonyms, remove this

source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate positive_only  

if [ $DATASET != "aap" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_positive_only.py \
                                                -dataset_path ./benchmarks/$DATASET \
                                                -model_name Positive_only-$DATASET \
                                                -model_load_path $MODEL_PATH \
                                                -model_save_path ./model/positive_only/"$DATASET"_"$DATATYPE" \
                                                -model_token_path facebook/bart-large \
                                                -logging_path ./logs/positive_only/$DATASET \
                                                -init_lr $LEARNING_RATE \
                                                -max_steps $STEPS \
                                                -warmup_steps 500 \
                                                -logging_steps 1000 \
                                                -per_device_train_batch_size $BATCH_SIZE \
                                                -seed 0 \
                                                -prefix_mention_is \
                                                -evaluation_strategy no \
                                                -lr_scheduler_type polynomial \
                                                -finetune \
                                                -dataset $DATATYPE                                    # If you want to don't use top-k synonyms, remove this

    CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_positive_only.py \
                                                -dataset_path ./benchmarks/$DATASET \
                                                -model_name Positive_only-$DATASET \
                                                -model_load_path ./model/positive_only/"$DATASET"_"$DATATYPE" \
                                                -model_token_path facebook/bart-large \
                                                -trie_path ./benchmarks/$DATASET/trie.pkl\
                                                -dict_path ./benchmarks/"$DATASET"/target_kb.json \
                                                -init_lr $LEARNING_RATE \
                                                -max_steps $STEPS \
                                                -per_device_train_batch_size $BATCH_SIZE \
                                                -per_device_eval_batch_size 1 \
                                                -seed 0 \
                                                -num_beams 10 \
                                                -prefix_prompt \
                                                -prefix_mention_is \
                                                -evaluation \
                                                -testset \
                                                -dataset $DATATYPE                                    # If you want to don't use top-k synonyms, remove this 

else
    for split in {0..9}; do
        CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_positive_only.py \
                                                    -dataset_path ./benchmarks/$DATASET/fold$split \
                                                    -model_name Positive_only-$DATASET \
                                                    -model_load_path $MODEL_PATH \
                                                    -model_save_path ./model/positive_only/"$DATASET"_"$DATATYPE"/fold$split \
                                                    -model_token_path facebook/bart-large \
                                                    -logging_path ./logs/$DATASET \
                                                    -init_lr $LEARNING_RATE \
                                                    -max_steps $STEPS \
                                                    -warmup_steps 500 \
                                                    -logging_steps 1000 \
                                                    -per_device_train_batch_size $BATCH_SIZE \
                                                    -seed 0 \
                                                    -prefix_mention_is \
                                                    -evaluation_strategy no \
                                                    -lr_scheduler_type polynomial \
                                                    -finetune \
                                                    -dataset $DATATYPE                                    # If you want to don't use top-k synonyms, remove this

        CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train_positive_only.py \
                                                    -dataset_path ./benchmarks/$DATASET/fold$split \
                                                    -model_name Positive_only-$DATASET \
                                                    -model_load_path ./model/positive_only/"$DATASET"_"$DATATYPE"/fold$split \
                                                    -model_token_path facebook/bart-large \
                                                    -trie_path ./benchmarks/$DATASET/trie.pkl\
                                                    -dict_path ./benchmarks/"$DATASET"/target_kb.json \
                                                    -init_lr $LEARNING_RATE \
                                                    -max_steps $STEPS \
                                                    -per_device_train_batch_size $BATCH_SIZE \
                                                    -per_device_eval_batch_size 1 \
                                                    -seed 0 \
                                                    -num_beams 10 \
                                                    -prefix_prompt \
                                                    -prefix_mention_is \
                                                    -evaluation \
                                                    -testset \
                                                    -dataset $DATATYPE                                    # If you want to don't use top-k synonyms, remove this
    done
fi
