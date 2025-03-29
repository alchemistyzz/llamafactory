export NPROC_PER_NODE=8
export MAX_PIXELS=1003520 
export WANDB_BASE_URL=https://api.wandb.ai 
export WANDB_PROJECT=cold_start 
export WANDB_API_KEY=19e2bb17296ca54e3b6de27ef184eac2eb7efd5f 
export WANDB_RUN_NAME=Qwen-VL2_5-7B-DPO-CRITIC-V-28K-$(date +%Y-%m-%d-%H-%M-%S) 
export USE_HF=1 


llamafactory-cli train script/dpo.yaml