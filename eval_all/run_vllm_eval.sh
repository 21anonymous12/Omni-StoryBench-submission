CUDA_VISIBLE_DEVICES=4,5 python eval_vllm.py \
  --text-backend vllm \
  --image-backend vllm \
  --json-repair-backend vllm \
  --text-batch-size 16 \
  --image-batch-size 4 \
  --vllm-tensor-parallel-size 2 \
  --text-vllm-max-model-len 8192 \
  --image-vllm-max-model-len 16384 2>&1 | tee log.txt
