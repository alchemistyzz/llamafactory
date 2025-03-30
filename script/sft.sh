export NPROC_PER_NODE=8
export MAX_PIXELS=1003520 
export WANDB_BASE_URL=https://api.wandb.ai 
export WANDB_PROJECT=cold_start 
export WANDB_API_KEY=19e2bb17296ca54e3b6de27ef184eac2eb7efd5f 
export WANDB_RUN_NAME=Qwen-VL2_5-7B-SFT-CRITIC-V-29K-$(date +%Y-%m-%d-%H-%M-%S) 
export USE_HF=1 

FORCE_TORCHRUN=1 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen2_vl \
    --flash_attn fa2 \
    --dataset_dir dataset \
    --dataset critique-VQA-SFT \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_steps 100 \
    --packing False \
    --report_to wandb \
    --output_dir outputs/Qwen2.5-VL-7B-Instruct/sft/train_$(date +%Y-%m-%d-%H-%M-%S) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_config.json 