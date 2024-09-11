DATASET=$1
# No model on huggingface for AskAPatient 

source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate positive_only  

python ./run_sample.py \
        -model_load_path dmis-lab/ANGEL_$DATASET \
        -model_token_path facebook/bart-large \
        -per_device_eval_batch_size 1 \
        -num_beams 5 \
        -prefix_mention_is \
