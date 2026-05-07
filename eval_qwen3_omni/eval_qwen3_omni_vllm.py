import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_CUDA_VISIBLE_DEVICES = "4"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if "/.local/lib/python" not in p]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_qwen3_omni_eval import batch_inference_qwen3_omni_eval
from src.vllm_utils import release_vllm_models


DEFAULT_DATASET_JSONL_PATH = Path(
    "The file path must be filled here."
)
DEFAULT_RESULTS_ROOT = Path(
    "The file path must be filled here."
)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "all_results.json"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "all_results_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate any-to-any results with a single "
            "Qwen3-Omni vLLM judge."
        )
    )
    parser.add_argument(
        "--dataset-jsonl-path",
        type=Path,
        default=DEFAULT_DATASET_JSONL_PATH,
        help="Path to dataset_speech_processed.jsonl.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing text/image/speech result folders.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Qwen3-Omni judge model name or local path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save per-sample evaluation results.",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="Path to save summary counts for the run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive dataset index to start from.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive dataset index to stop at. Defaults to the dataset length.",
    )
    parser.add_argument(
        "--checkpoint-start-index",
        type=int,
        default=5,
        help="First dataset index at which to save an intermediate checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint interval in dataset-index units after checkpoint-start-index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for Qwen3-Omni judge requests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum judge response tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value for the judge.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the Qwen3-Omni vLLM engine.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization target for the vLLM engine.",
    )
    parser.add_argument(
        "--vllm-swap-space",
        type=float,
        default=8.0,
        help="Swap space in GiB for the vLLM engine.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=32768,
        help="Max model length for the Qwen3-Omni vLLM engine.",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=1,
        help="Max concurrent sequences for the Qwen3-Omni vLLM engine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="vLLM random seed.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Forward use_audio_in_video=True to Qwen3-Omni preprocessing.",
    )
    return parser.parse_args()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_missing_reason(sample_dir: Path) -> Optional[str]:
    missing_path = sample_dir / "missing.txt"
    if missing_path.is_file():
        return read_text_file(missing_path) or f"Missing candidate artifact in {sample_dir}"
    return None


def build_missing_candidate(
    modality: str,
    sample_dir: Path,
    default_reason: str,
) -> Dict[str, Any]:
    return {
        "status": "missing",
        "sample_dir": str(sample_dir),
        "path": None,
        "missing_reason": read_missing_reason(sample_dir) or default_reason,
        "modality": modality,
    }


def natural_sort_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_candidate_files(
    sample_dir: Path,
    *,
    preferred_names: Sequence[str],
    patterns: Sequence[str],
    excluded_names: Sequence[str] = (),
    excluded_name_tokens: Sequence[str] = (),
) -> List[Path]:
    excluded = set(excluded_names)
    lowered_tokens = [token.lower() for token in excluded_name_tokens]
    candidates: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        lowered_name = path.name.lower()
        if not path.is_file() or path.name in excluded or path in seen:
            return
        if any(token in lowered_name for token in lowered_tokens):
            return
        candidates.append(path)
        seen.add(path)

    for name in preferred_names:
        add(sample_dir / name)
    for pattern in patterns:
        for path in sorted(sample_dir.glob(pattern), key=natural_sort_key):
            add(path)

    return candidates


def resolve_text_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "text" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "text",
            sample_dir,
            f"Text sample directory does not exist: {sample_dir}",
        )
    text_candidates = list_candidate_files(
        sample_dir,
        preferred_names=[
            "generated.txt",
            "generated_text.txt",
            "text.txt",
            "candidate.txt",
            "output.txt",
        ],
        patterns=[
            "generated_text_*.txt",
            "generated_[0-9]*.txt",
            "text_[0-9]*.txt",
            "candidate*.txt",
            "output*.txt",
            "*.txt",
        ],
        excluded_names=[
            "missing.txt",
            "generated_speech_text.txt",
            "raw_text_response.txt",
            "text_prompt.txt",
            "prompt.txt",
        ],
        excluded_name_tokens=["prompt", "raw", "response", "speech"],
    )
    if not text_candidates:
        return build_missing_candidate(
            "text",
            sample_dir,
            f"Text candidate file not found in {sample_dir}",
        )

    generated_path = text_candidates[0]
    text = read_text_file(generated_path)
    if not text:
        return {
            "status": "empty",
            "sample_dir": str(sample_dir),
            "path": str(generated_path),
            "all_candidate_paths": [str(path) for path in text_candidates],
            "text": "",
            "missing_reason": "Text candidate file is empty.",
            "modality": "text",
        }

    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(generated_path),
        "all_candidate_paths": [str(path) for path in text_candidates],
        "text": text,
        "modality": "text",
    }


def resolve_image_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "image" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "image",
            sample_dir,
            f"Image sample directory does not exist: {sample_dir}",
        )
    image_candidates = list_candidate_files(
        sample_dir,
        preferred_names=["generated.png"],
        patterns=[
            "generated_*.png",
            "generated_*.jpg",
            "generated_*.jpeg",
            "generated_*.webp",
            "image_*.png",
            "image_*.jpg",
            "image_*.jpeg",
            "image_*.webp",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.webp",
        ],
    )
    if not image_candidates:
        return build_missing_candidate(
            "image",
            sample_dir,
            f"Image candidate file not found in {sample_dir}",
        )

    generated_path = image_candidates[0]
    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(generated_path),
        "all_candidate_paths": [str(path) for path in image_candidates],
        "modality": "image",
    }


def resolve_speech_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "speech" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "speech",
            sample_dir,
            f"Speech sample directory does not exist: {sample_dir}",
        )

    audio_candidates = list_candidate_files(
        sample_dir,
        preferred_names=["generated.wav"],
        patterns=[
            "generated_*.wav",
            "generated_*.flac",
            "generated_*.mp3",
            "generated_*.ogg",
            "generated_*.m4a",
            "speech_*.wav",
            "speech_*.flac",
            "speech_*.mp3",
            "speech_*.ogg",
            "speech_*.m4a",
            "*.wav",
            "*.flac",
            "*.mp3",
            "*.ogg",
            "*.m4a",
        ],
    )

    if not audio_candidates:
        return build_missing_candidate(
            "speech",
            sample_dir,
            f"Speech candidate audio file not found in {sample_dir}",
        )

    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(audio_candidates[0]),
        "all_candidate_paths": [str(path) for path in audio_candidates],
        "all_audio_paths": [str(path) for path in audio_candidates],
        "modality": "speech",
    }


def extract_ground_truth_speech_text(record: Dict[str, Any]) -> str:
    speech_metadata = record.get("speech_metadata_next_page", {})
    parsed_metadata = (
        speech_metadata.get("parsed", speech_metadata)
        if isinstance(speech_metadata, dict)
        else speech_metadata
    )
    if isinstance(parsed_metadata, dict):
        speaker = parsed_metadata.get("speaker")
        if isinstance(speaker, dict):
            line = speaker.get("line")
            if isinstance(line, str) and line.strip():
                return line.strip()

        lines: List[str] = []
        for key, value in parsed_metadata.items():
            if not key.startswith("speaker_") or not isinstance(value, dict):
                continue
            line = value.get("line")
            if isinstance(line, str) and line.strip():
                lines.append(line.strip())
        if lines:
            return " ".join(lines)

        line = parsed_metadata.get("line")
        if isinstance(line, str) and line.strip():
            return line.strip()

    next_page = record.get("next_page", {})
    for key in ("text_processed", "text_speech_sep", "text"):
        value = next_page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def build_vllm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "tensor_parallel_size": max(args.vllm_tensor_parallel_size, 1),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "max_num_seqs": max(args.vllm_max_num_seqs, args.batch_size),
        "swap_space": args.vllm_swap_space,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {
            "image": 3,
            "audio": 1,
            "video": 0,
        },
        "seed": args.seed,
    }


def chunked(values: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def prepare_record_entry(
    dataset_index: int,
    progress_index: int,
    total_records: int,
    record: Dict[str, Any],
    results_root: Path,
) -> Dict[str, Any]:
    source = record.get("source", "unknown")
    book = record.get("book", "unknown")
    prev_key = record.get("prev_key", "unknown")
    next_key = record.get("next_key", "unknown")

    print(
        f"[prepare {progress_index}/{total_records}] dataset_index={dataset_index} "
        f"{source}/{book} {prev_key} -> {next_key}"
    )

    return {
        "dataset_index": dataset_index,
        "record": record,
        "candidate": {
            "text": resolve_text_candidate(results_root, dataset_index),
            "image": resolve_image_candidate(results_root, dataset_index),
            "speech": resolve_speech_candidate(results_root, dataset_index),
        },
    }


def build_base_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    record = entry["record"]
    return {
        "dataset_index": entry["dataset_index"],
        "record_status": "pending",
        "source": record.get("source", "unknown"),
        "book": record.get("book", "unknown"),
        "prev_key": record.get("prev_key", "unknown"),
        "next_key": record.get("next_key", "unknown"),
        "metadata": record.get("metadata", {}),
        "next_page_condition": record.get("next_page_condition", {}),
        "current_page": record.get("current_page", {}),
        "next_page": record.get("next_page", {}),
        "ground_truth_speech_text": extract_ground_truth_speech_text(record),
        "ground_truth_speech_metadata": record.get("speech_metadata_next_page", {}),
        "candidate": entry["candidate"],
        "omni": None,
    }


def missing_candidate_modalities(entry: Dict[str, Any]) -> List[str]:
    return [
        modality
        for modality, candidate in entry["candidate"].items()
        if candidate.get("status") != "ok"
    ]


def build_skipped_omni_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    missing = missing_candidate_modalities(entry)
    reasons = {
        modality: entry["candidate"][modality].get(
            "missing_reason",
            f"{modality} candidate is unavailable.",
        )
        for modality in missing
    }
    return {
        "status": "skipped",
        "reason": "At least one required candidate modality is unavailable.",
        "missing_modalities": missing,
        "missing_reasons": reasons,
        "qwen3_omni_eval": None,
        "raw_response": None,
        "parse_error": None,
    }


def finalize_record_status(result: Dict[str, Any]) -> str:
    if result.get("record_status") == "error":
        return "error"

    candidate_summary = result.get("candidate", {})
    if any(candidate.get("status") != "ok" for candidate in candidate_summary.values()):
        return "skipped_missing"

    omni_result = result.get("omni") or {}
    omni_status = omni_result.get("status")
    if omni_status == "ok":
        return "ok"
    if omni_status == "parse_error":
        return "parse_error"
    if omni_status == "skipped":
        return "skipped"
    return "error"


def build_omni_request(entry: Dict[str, Any]) -> Dict[str, Any]:
    record = entry["record"]
    current_page = record.get("current_page", {})
    next_page = record.get("next_page", {})
    candidate = entry["candidate"]

    return {
        "metadata": record.get("metadata", {}),
        "condition_json": record.get("next_page_condition", {}),
        "current_text": current_page.get("text", ""),
        "ground_truth_text": next_page.get("text", ""),
        "candidate_text": candidate["text"]["text"],
        "ground_truth_speech_metadata": record.get("speech_metadata_next_page", {}),
        "current_image": current_page.get("image_path", ""),
        "ground_truth_image": next_page.get("image_path", ""),
        "candidate_image": candidate["image"]["path"],
        "candidate_audio": candidate["speech"]["path"],
    }


def evaluate_omni_batch(
    entries: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    vllm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    start_index = entries[0]["dataset_index"]
    end_index = entries[-1]["dataset_index"]
    print(
        f"[omni batch] evaluating {len(entries)} sample(s) "
        f"dataset_index={start_index}..{end_index}"
    )

    requests = [build_omni_request(entry) for entry in entries]
    try:
        return batch_inference_qwen3_omni_eval(
            args.model_name,
            requests,
            vllm_kwargs=vllm_kwargs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_audio_in_video=args.use_audio_in_video,
        )
    except Exception as batch_exc:
        print(
            f"[Warning] Omni batch failed with {type(batch_exc).__name__}: {batch_exc}. "
            "Falling back to single-request evaluation."
        )

    results: List[Dict[str, Any]] = []
    for request in requests:
        try:
            results.append(
                batch_inference_qwen3_omni_eval(
                    args.model_name,
                    [request],
                    vllm_kwargs=vllm_kwargs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_audio_in_video=args.use_audio_in_video,
                )[0]
            )
        except Exception as single_exc:
            results.append(
                {
                    "status": "error",
                    "reason": str(single_exc),
                    "error_type": type(single_exc).__name__,
                    "qwen3_omni_eval": None,
                    "raw_response": None,
                    "parse_error": None,
                }
            )
    return results


def build_error_result(
    dataset_index: int,
    record: Dict[str, Any],
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "dataset_index": dataset_index,
        "record_status": "error",
        "source": record.get("source", "unknown"),
        "book": record.get("book", "unknown"),
        "prev_key": record.get("prev_key", "unknown"),
        "next_key": record.get("next_key", "unknown"),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def build_summary(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    total_dataset_records: int,
    start_index: int,
    end_index: int,
) -> Dict[str, Any]:
    record_status_counts = Counter(result.get("record_status", "unknown") for result in results)
    candidate_status_counts = {
        modality: Counter(
            result.get("candidate", {}).get(modality, {}).get("status", "unknown")
            for result in results
        )
        for modality in ("text", "image", "speech")
    }
    omni_status_counts = Counter(
        result.get("omni", {}).get("status", "unknown")
        for result in results
    )
    omni_error_type_counts = Counter(
        result.get("omni", {}).get("error_type", "not_available")
        for result in results
        if result.get("omni", {}).get("status") in {"parse_error", "error"}
    )

    return {
        "dataset_jsonl_path": str(args.dataset_jsonl_path),
        "results_root": str(args.results_root),
        "output_path": str(args.output_path),
        "summary_output_path": str(args.summary_output_path),
        "model_name": args.model_name,
        "judge_backend": "vllm",
        "batch_size": args.batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_audio_in_video": args.use_audio_in_video,
        "total_dataset_records": total_dataset_records,
        "evaluated_record_count": len(results),
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "record_status_counts": dict(record_status_counts),
        "candidate_status_counts": {
            modality: dict(counts)
            for modality, counts in candidate_status_counts.items()
        },
        "omni_status_counts": dict(omni_status_counts),
        "omni_error_type_counts": dict(omni_error_type_counts),
    }


def save_results_snapshot(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    total_dataset_records: int,
    start_index: int,
    end_index: int,
    *,
    checkpoint_label: Optional[str] = None,
) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    summary = build_summary(
        results,
        args,
        total_dataset_records=total_dataset_records,
        start_index=start_index,
        end_index=end_index,
    )
    if checkpoint_label is not None:
        summary["checkpoint_label"] = checkpoint_label

    args.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def should_save_checkpoint(
    dataset_index: int,
    checkpoint_start_index: int,
    checkpoint_every: int,
) -> bool:
    if checkpoint_every <= 0:
        return False
    if dataset_index < checkpoint_start_index:
        return False
    return (dataset_index - checkpoint_start_index) % checkpoint_every == 0


def append_result_with_checkpoint(
    all_results: List[Dict[str, Any]],
    result: Dict[str, Any],
    dataset_index: int,
    args: argparse.Namespace,
    total_dataset_records: int,
    start_index: int,
    end_index: int,
) -> None:
    all_results.append(result)
    if should_save_checkpoint(
        dataset_index,
        args.checkpoint_start_index,
        args.checkpoint_every,
    ):
        checkpoint_label = f"dataset_index_{dataset_index}"
        save_results_snapshot(
            all_results,
            args,
            total_dataset_records=total_dataset_records,
            start_index=start_index,
            end_index=end_index,
            checkpoint_label=checkpoint_label,
        )
        print(
            f"[checkpoint] Saved intermediate results at dataset_index={dataset_index} "
            f"to {args.output_path}"
        )


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset_jsonl_path)
    vllm_kwargs = build_vllm_kwargs(args)

    start_index = max(args.start_index, 0)
    end_index = len(records) if args.end_index is None else min(args.end_index, len(records))
    if start_index >= end_index:
        raise ValueError(
            f"Invalid evaluation range: start_index={start_index}, end_index={end_index}"
        )

    total_selected_records = end_index - start_index
    all_results: List[Dict[str, Any]] = []
    pending_entries: List[Dict[str, Any]] = []

    print(
        f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"model={args.model_name} "
        f"batch_size={args.batch_size} tensor_parallel_size={args.vllm_tensor_parallel_size}"
    )

    def flush_pending() -> None:
        nonlocal pending_entries
        if not pending_entries:
            return
        batch_results = evaluate_omni_batch(pending_entries, args, vllm_kwargs)
        for entry, omni_result in zip(pending_entries, batch_results):
            result = build_base_result(entry)
            result["omni"] = omni_result
            result["record_status"] = finalize_record_status(result)
            append_result_with_checkpoint(
                all_results,
                result,
                entry["dataset_index"],
                args,
                total_dataset_records=len(records),
                start_index=start_index,
                end_index=end_index,
            )
        pending_entries = []

    try:
        for progress_index, dataset_index in enumerate(range(start_index, end_index), start=1):
            record = records[dataset_index]
            try:
                entry = prepare_record_entry(
                    dataset_index=dataset_index,
                    progress_index=progress_index,
                    total_records=total_selected_records,
                    record=record,
                    results_root=args.results_root,
                )
                if missing_candidate_modalities(entry):
                    flush_pending()
                    result = build_base_result(entry)
                    result["omni"] = build_skipped_omni_result(entry)
                    result["record_status"] = finalize_record_status(result)
                    append_result_with_checkpoint(
                        all_results,
                        result,
                        dataset_index,
                        args,
                        total_dataset_records=len(records),
                        start_index=start_index,
                        end_index=end_index,
                    )
                    continue

                pending_entries.append(entry)
                if len(pending_entries) >= args.batch_size:
                    flush_pending()
            except Exception as error:
                flush_pending()
                print(
                    f"[prepare/eval failure] dataset_index={dataset_index} "
                    f"{type(error).__name__}: {error}"
                )
                append_result_with_checkpoint(
                    all_results,
                    build_error_result(dataset_index, record, error),
                    dataset_index,
                    args,
                    total_dataset_records=len(records),
                    start_index=start_index,
                    end_index=end_index,
                )

        flush_pending()

        save_results_snapshot(
            all_results,
            args,
            total_dataset_records=len(records),
            start_index=start_index,
            end_index=end_index,
        )
    finally:
        release_vllm_models(model_name=args.model_name)


if __name__ == "__main__":
    main()
