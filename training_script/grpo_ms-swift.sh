# 设置共享内存路径和大小
export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_MODE=offline
export TMPDIR=/tmp
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
sudo mount -o remount,size=64G /dev/shm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAX_PIXELS=262144 \
MASTER_PORT=29600 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /path/to/your/model \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --model_type qwen2_5_vl \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.85 \
    --train_type full \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --dataset /path/to/your/dataset \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-7 \
    --eval_steps 10000 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output/your-model-name-grpo-training \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate false \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --offload_optimizer true \
    --offload_model true \
    --report_to wandb \
    --swanlab_project your-project-name \
    --gc_collect_after_offload true
