CUDA_VISIBLE_DEVICES=4 python eval_qwen3_omni_vllm.py \
  --model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --batch-size 1 \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-model-len 32768 \
  --vllm-max-num-seqs 1 2>&1 | tee log.txt
