export DS_SKIP_CUDA_CHECK="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_AIO=1
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9


deepspeed --include localhost:0 \
          ./train.py --config-file ./bart.json \
                     --output_dir ./synonyms_pretrained_model \
                     --token_nosing_prob 0.1 \
		             --label_smoothing_factor 0.1 \
                     --max_seq_length 1024 \
                     --max_predictions_per_seq 150 \
                     --seed 42 \
                     --lr_schedule LL \
                     --job_name st21pv_pretrain \
                     --print_steps 10 \
                     --save_steps 100 \
                     --data_path_prefix ./datagen \
                     --deepspeed \
                     --deepspeed_config ./ds_config_zero2.json
