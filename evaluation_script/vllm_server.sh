## model path填写模型地址##
## model name填写模型名字##
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve ${model path}  \
  --trust-remote-code \
  --served-model-name ${model name} \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 4 \
  --port 8999 \
  --max-parallel-loading-workers 8 \
