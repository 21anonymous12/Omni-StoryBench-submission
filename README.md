# Omni-StoryBench Evaluation Code

This repository contains the evaluation scripts used for story-grounded omnimodal generation experiments. All local data paths are intentionally omitted for anonymous review. Replace each placeholder path with paths in your own environment before running the scripts.

## Contents

- `eval_all/`: evaluates text, image, and speech outputs with separate judges and automatic metrics.
- `eval_qwen3_omni/`: evaluates text, image, and speech outputs with a single Qwen3-Omni judge.
- `eval_speech_metadata/`: extracts speech metadata from generated audio and compares it with benchmark metadata labels.

## Expected Inputs

The main evaluators expect:

- a benchmark JSONL file passed with `--dataset-jsonl-path` or `--gt-jsonl`;
- a candidate results root passed with `--results-root`;
- generated outputs organized by modality and dataset index:

```text
<results-root>/
  text/<index>/
  image/<index>/
  speech/<index>/
```

Text folders may contain files such as `generated.txt`, `candidate.txt`, or other `.txt` outputs. Image folders may contain `.png`, `.jpg`, `.jpeg`, or `.webp` outputs. Speech folders may contain `.wav`, `.flac`, `.mp3`, `.ogg`, or `.m4a` outputs.

The benchmark JSONL records should include the fields consumed by the scripts, including `metadata`, `next_page_condition`, `current_page`, `next_page`, and `speech_metadata_next_page`.

## Environment

Use a Python environment with the model and evaluation dependencies installed:

```bash
pip install torch transformers vllm pillow bert-score librosa numpy huggingface_hub speechbrain funasr openai-whisper qwen-omni-utils
```

Some dependencies are only required for specific evaluators or backends. Install the versions compatible with your GPU, CUDA, PyTorch, and vLLM setup.

## Run Separate Text/Image/Speech Evaluation

```bash
cd eval_all
python eval_vllm.py \
  --dataset-jsonl-path "The file path must be filled here." \
  --results-root "The file path must be filled here." \
  --output-path all_results.json \
  --summary-output-path all_results_summary.json
```

Useful backend options:

```bash
python eval_vllm.py \
  --dataset-jsonl-path "The file path must be filled here." \
  --results-root "The file path must be filled here." \
  --text-backend vllm \
  --image-backend vllm \
  --json-repair-backend vllm \
  --text-batch-size 16 \
  --image-batch-size 4
```

## Run Qwen3-Omni Evaluation

```bash
cd eval_qwen3_omni
python eval_qwen3_omni_vllm.py \
  --dataset-jsonl-path "The file path must be filled here." \
  --results-root "The file path must be filled here." \
  --output-path all_results.json \
  --summary-output-path all_results_summary.json
```

## Run Speech Metadata Evaluation

```bash
cd eval_speech_metadata
python evaluate_speech_metadata.py \
  --speech-root "The file path must be filled here." \
  --gt-jsonl "The file path must be filled here." \
  --output-dir .
```

This writes `metadata_comparison.json` and `class_frequency_summary.json` by default. Output paths are restricted to the `eval_speech_metadata/` directory unless the script is modified.

## Notes

- The default path placeholders must be replaced through command-line arguments.
- Adjust `CUDA_VISIBLE_DEVICES`, batch sizes, tensor parallel settings, and model names for your own compute environment.
- The scripts write JSON result files and optional logs in the directory where they are run.
