export N_GPUS=2
export BASE_MODEL=/root/autodl-tmp/models/llm/Qwen2.5-3B
export DATA_DIR=/root/autodl-tmp/code/wangzhengzhuo512/TinyZero/data/invest
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=invest-qwen2.5-3B
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=108f1dfd2e46cdcbb2165b1379943e38da8dbbc7
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash ./scripts/train_tiny_zero.sh

# pip3 install nvidia-cublas-cu12==12.3.4.1 cl




