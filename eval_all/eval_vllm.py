import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_CUDA_VISIBLE_DEVICES = "4,5"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if "/.local/lib/python" not in p]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bertscore import clear_bert_score_cache, get_bert_scores
from src.clip_similarity import clear_clip_model_cache, clip_similarity_batch
from src.inference_qwen3_eval_image import batch_inference_qwen3_eval_image
from src.inference_qwen3_eval_speech import inference_qwen3_eval_speech
from src.inference_qwen3_eval_text import batch_inference_qwen3_eval_text
from src.vllm_utils import release_vllm_models, should_enable_expert_parallel


DEFAULT_DATASET_JSONL_PATH = Path(
    "The file path must be filled here."
)
DEFAULT_RESULTS_ROOT = Path(
    "The file path must be filled here."
)
DEFAULT_TEXT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_IMAGE_MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_SPEECH_MODEL_NAME = "nvidia/audio-flamingo-next-hf"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "all_results.json"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "all_results_summary.json"
DEFAULT_TEXT_BACKEND = "auto"
DEFAULT_IMAGE_BACKEND = "auto"
DEFAULT_JSON_REPAIR_BACKEND = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate any-to-any results against the benchmark."
    )
    parser.add_argument(
        "--dataset-jsonl-path",
        type=Path,
        default=DEFAULT_DATASET_JSONL_PATH,
        help="Path to dataset_speech_processed.jsonl",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing text/image/speech result folders.",
    )
    parser.add_argument(
        "--text-model-name",
        type=str,
        default=DEFAULT_TEXT_MODEL_NAME,
        help="LLM-as-a-judge model for text evaluation.",
    )
    parser.add_argument(
        "--image-model-name",
        type=str,
        default=DEFAULT_IMAGE_MODEL_NAME,
        help="VLM-as-a-judge model for image evaluation.",
    )
    parser.add_argument(
        "--speech-model-name",
        type=str,
        default=DEFAULT_SPEECH_MODEL_NAME,
        help="Audio judge model for speech evaluation.",
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
        "--text-backend",
        type=str,
        choices=["auto", "transformers", "vllm"],
        default=DEFAULT_TEXT_BACKEND,
        help="Backend for text judge inference.",
    )
    parser.add_argument(
        "--image-backend",
        type=str,
        choices=["auto", "transformers", "vllm"],
        default=DEFAULT_IMAGE_BACKEND,
        help="Backend for image judge inference.",
    )
    parser.add_argument(
        "--json-repair-backend",
        type=str,
        choices=["auto", "transformers", "vllm"],
        default=DEFAULT_JSON_REPAIR_BACKEND,
        help="Backend for speech JSON repair with Qwen.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=16,
        help="Batch size for text judge requests.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=4,
        help="Batch size for image judge requests.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM-backed judges.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization target for vLLM engines.",
    )
    parser.add_argument(
        "--vllm-swap-space",
        type=float,
        default=8.0,
        help="Swap space in GiB for vLLM engines.",
    )
    parser.add_argument(
        "--text-vllm-max-model-len",
        type=int,
        default=8192,
        help="Max model length for the text judge vLLM engine.",
    )
    parser.add_argument(
        "--image-vllm-max-model-len",
        type=int,
        default=16384,
        help="Max model length for the image judge vLLM engine.",
    )
    parser.add_argument(
        "--text-vllm-max-num-seqs",
        type=int,
        default=16,
        help="Max concurrent sequences for the text judge vLLM engine.",
    )
    parser.add_argument(
        "--image-vllm-max-num-seqs",
        type=int,
        default=4,
        help="Max concurrent sequences for the image judge vLLM engine.",
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


def resolve_text_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "text" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "text",
            sample_dir,
            f"Text sample directory does not exist: {sample_dir}",
        )

    text_candidates: List[Path] = []
    seen_paths = set()
    for pattern in ("page_*.txt", "generated*.txt", "*.txt"):
        for path in sorted(sample_dir.glob(pattern)):
            if (
                path.is_file()
                and path.name != "missing.txt"
                and path not in seen_paths
            ):
                text_candidates.append(path)
                seen_paths.add(path)

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
            "text": "",
            "missing_reason": "Text candidate file is empty.",
        }

    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(generated_path),
        "text": text,
    }


def resolve_image_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "image" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "image",
            sample_dir,
            f"Image sample directory does not exist: {sample_dir}",
        )

    image_candidates: List[Path] = []
    seen_paths = set()
    for pattern in (
        "page_*.png",
        "page_*.jpg",
        "page_*.jpeg",
        "generated*.png",
        "generated*.jpg",
        "generated*.jpeg",
        "*.png",
        "*.jpg",
        "*.jpeg",
    ):
        for path in sorted(sample_dir.glob(pattern)):
            if path.is_file() and path not in seen_paths:
                image_candidates.append(path)
                seen_paths.add(path)

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
    }


def resolve_speech_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "speech" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "speech",
            sample_dir,
            f"Speech sample directory does not exist: {sample_dir}",
        )

    audio_candidates: List[Path] = []
    seen_paths = set()
    for pattern in ("page_*.wav", "generated_*.wav", "generated.wav", "*.wav"):
        for path in sorted(sample_dir.glob(pattern)):
            if path.is_file() and path not in seen_paths:
                audio_candidates.append(path)
                seen_paths.add(path)

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
        "all_audio_paths": [str(path) for path in audio_candidates],
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


def build_text_vllm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "tensor_parallel_size": max(args.vllm_tensor_parallel_size, 1),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.text_vllm_max_model_len,
        "max_num_seqs": max(args.text_vllm_max_num_seqs, args.text_batch_size),
        "swap_space": args.vllm_swap_space,
        "distributed_executor_backend": "mp",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enable_expert_parallel": (
            True if should_enable_expert_parallel(args.text_model_name) else None
        ),
    }


def build_image_vllm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "tensor_parallel_size": max(args.vllm_tensor_parallel_size, 1),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.image_vllm_max_model_len,
        "max_num_seqs": max(args.image_vllm_max_num_seqs, args.image_batch_size),
        "swap_space": args.vllm_swap_space,
        "distributed_executor_backend": "mp",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": 3, "video": 0},
        "enable_expert_parallel": (
            True if should_enable_expert_parallel(args.image_model_name) else None
        ),
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
        "candidate": entry["candidate"],
        "text": None,
        "image": None,
        "speech": None,
    }


def finalize_record_status(result: Dict[str, Any]) -> str:
    candidate_summary = result.get("candidate", {})
    text_result = result.get("text") or {}
    image_result = result.get("image") or {}
    speech_result = result.get("speech") or {}

    record_status = "ok"
    if any(item.get("status") != "ok" for item in candidate_summary.values()):
        record_status = "partial_missing"
    if any(
        modality_result.get("status") == "error"
        for modality_result in (text_result, image_result, speech_result)
    ):
        record_status = "partial_error"
    return record_status


def evaluate_text_entries(
    entries: Sequence[Dict[str, Any]],
    model_name: str,
    *,
    backend: str,
    vllm_kwargs: Dict[str, Any],
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    valid_entries: List[Dict[str, Any]] = []

    for entry in entries:
        candidate_text = entry["candidate"]["text"]
        if candidate_text["status"] != "ok":
            results[entry["dataset_index"]] = {
                "status": "skipped",
                "reason": candidate_text.get("missing_reason", "Text candidate is unavailable."),
                "bert_score": None,
                "qwen3_text_eval": None,
            }
            continue
        valid_entries.append(entry)

    for batch in chunked(valid_entries, batch_size):
        start_index = batch[0]["dataset_index"]
        end_index = batch[-1]["dataset_index"]
        print(
            f"[text batch] evaluating {len(batch)} sample(s) "
            f"dataset_index={start_index}..{end_index}"
        )

        predictions = [entry["candidate"]["text"]["text"] for entry in batch]
        ground_truths = [
            entry["record"].get("next_page", {}).get("text", "") for entry in batch
        ]
        _, _, bert_scores = get_bert_scores(predictions, ground_truths)
        requests = [
            {
                "metadata": entry["record"].get("metadata", {}),
                "condition_json": entry["record"].get("next_page_condition", {}),
                "current_text": entry["record"].get("current_page", {}).get("text", ""),
                "ground_truth_text": entry["record"].get("next_page", {}).get("text", ""),
                "candidate_text": entry["candidate"]["text"]["text"],
            }
            for entry in batch
        ]

        try:
            judge_results = batch_inference_qwen3_eval_text(
                model_name,
                requests,
                backend=backend,
                vllm_kwargs=vllm_kwargs,
            )
        except Exception as batch_exc:
            print(
                f"[Warning] Text batch failed with {type(batch_exc).__name__}: {batch_exc}. "
                "Falling back to single-request evaluation."
            )
            judge_results = []
            for request in requests:
                try:
                    judge_results.append(
                        batch_inference_qwen3_eval_text(
                            model_name,
                            [request],
                            backend=backend,
                            vllm_kwargs=vllm_kwargs,
                        )[0]
                    )
                except Exception as single_exc:
                    judge_results.append(
                        {
                            "error": str(single_exc),
                            "error_type": type(single_exc).__name__,
                        }
                    )

        for entry, bert_score, judge_result in zip(batch, bert_scores, judge_results):
            if isinstance(judge_result, dict) and "error" in judge_result:
                results[entry["dataset_index"]] = {
                    "status": "error",
                    "reason": judge_result["error"],
                    "error_type": judge_result.get("error_type", "TextJudgeError"),
                    "bert_score": bert_score,
                    "qwen3_text_eval": None,
                }
                continue

            results[entry["dataset_index"]] = {
                "status": "ok",
                "bert_score": bert_score,
                "qwen3_text_eval": judge_result,
            }

    return results


def evaluate_image_entries(
    entries: Sequence[Dict[str, Any]],
    model_name: str,
    *,
    backend: str,
    vllm_kwargs: Dict[str, Any],
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    valid_entries: List[Dict[str, Any]] = []

    for entry in entries:
        candidate_image = entry["candidate"]["image"]
        if candidate_image["status"] != "ok":
            results[entry["dataset_index"]] = {
                "status": "skipped",
                "reason": candidate_image.get(
                    "missing_reason", "Image candidate is unavailable."
                ),
                "clip_similarity": None,
                "qwen3_image_eval": None,
            }
            continue
        valid_entries.append(entry)

    for batch in chunked(valid_entries, batch_size):
        start_index = batch[0]["dataset_index"]
        end_index = batch[-1]["dataset_index"]
        print(
            f"[image batch] evaluating {len(batch)} sample(s) "
            f"dataset_index={start_index}..{end_index}"
        )

        clip_pairs = [
            (
                entry["record"].get("next_page", {}).get("image_path", ""),
                entry["candidate"]["image"]["path"],
            )
            for entry in batch
        ]
        try:
            clip_scores = clip_similarity_batch(clip_pairs)
            clip_errors: List[Optional[str]] = [None] * len(batch)
        except Exception as clip_batch_exc:
            print(
                f"[Warning] CLIP batch failed with {type(clip_batch_exc).__name__}: {clip_batch_exc}. "
                "Falling back to single-pair similarity."
            )
            clip_scores = []
            clip_errors = []
            for pair in clip_pairs:
                try:
                    clip_scores.append(clip_similarity_batch([pair])[0])
                    clip_errors.append(None)
                except Exception as single_clip_exc:
                    clip_scores.append(None)
                    clip_errors.append(str(single_clip_exc))

        requests = [
            {
                "metadata": entry["record"].get("metadata", {}),
                "condition_json": entry["record"].get("next_page_condition", {}),
                "current_image": entry["record"].get("current_page", {}).get("image_path", ""),
                "ground_truth_image": entry["record"].get("next_page", {}).get("image_path", ""),
                "candidate_image": entry["candidate"]["image"]["path"],
            }
            for entry in batch
        ]

        try:
            judge_results = batch_inference_qwen3_eval_image(
                model_name,
                requests,
                backend=backend,
                vllm_kwargs=vllm_kwargs,
            )
        except Exception as batch_exc:
            print(
                f"[Warning] Image batch failed with {type(batch_exc).__name__}: {batch_exc}. "
                "Falling back to single-request evaluation."
            )
            judge_results = []
            for request in requests:
                try:
                    judge_results.append(
                        batch_inference_qwen3_eval_image(
                            model_name,
                            [request],
                            backend=backend,
                            vllm_kwargs=vllm_kwargs,
                        )[0]
                    )
                except Exception as single_exc:
                    judge_results.append(
                        {
                            "error": str(single_exc),
                            "error_type": type(single_exc).__name__,
                        }
                    )

        for entry, clip_score, clip_error, judge_result in zip(
            batch, clip_scores, clip_errors, judge_results
        ):
            if clip_error is not None:
                results[entry["dataset_index"]] = {
                    "status": "error",
                    "reason": clip_error,
                    "error_type": "CLIPSimilarityError",
                    "clip_similarity": None,
                    "qwen3_image_eval": judge_result
                    if not (isinstance(judge_result, dict) and "error" in judge_result)
                    else None,
                }
                continue

            if isinstance(judge_result, dict) and "error" in judge_result:
                results[entry["dataset_index"]] = {
                    "status": "error",
                    "reason": judge_result["error"],
                    "error_type": judge_result.get("error_type", "ImageJudgeError"),
                    "clip_similarity": clip_score,
                    "qwen3_image_eval": None,
                }
                continue

            results[entry["dataset_index"]] = {
                "status": "ok",
                "clip_similarity": clip_score,
                "qwen3_image_eval": judge_result,
            }

    return results


def evaluate_speech_modality(
    model_name: str,
    record: Dict[str, Any],
    candidate_speech: Dict[str, Any],
    *,
    json_repair_backend: str,
    vllm_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    ground_truth_speech_metadata = record.get("speech_metadata_next_page", {})
    ground_truth_speech_text = extract_ground_truth_speech_text(record)

    if candidate_speech["status"] != "ok":
        return {
            "status": "skipped",
            "reason": candidate_speech.get(
                "missing_reason",
                "Speech candidate audio is unavailable.",
            ),
            "ground_truth_speech_text": ground_truth_speech_text,
            "ground_truth_speech_metadata": ground_truth_speech_metadata,
            "audio_flamingo_next_eval": None,
        }

    try:
        audio_eval_result = inference_qwen3_eval_speech(
            model_name,
            record.get("metadata", {}),
            record.get("next_page_condition", {}),
            ground_truth_speech_metadata,
            candidate_speech["path"],
            json_repair_backend=json_repair_backend,
            vllm_kwargs=vllm_kwargs,
        )
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "ground_truth_speech_text": ground_truth_speech_text,
            "ground_truth_speech_metadata": ground_truth_speech_metadata,
            "audio_flamingo_next_eval": None,
        }

    if isinstance(audio_eval_result, dict) and "error" in audio_eval_result:
        return {
            "status": "error",
            "reason": audio_eval_result["error"],
            "error_type": "SpeechJudgeJSONParseError",
            "ground_truth_speech_text": ground_truth_speech_text,
            "ground_truth_speech_metadata": ground_truth_speech_metadata,
            "audio_flamingo_next_eval": audio_eval_result,
        }

    return {
        "status": "ok",
        "ground_truth_speech_text": ground_truth_speech_text,
        "ground_truth_speech_metadata": ground_truth_speech_metadata,
        "audio_flamingo_next_eval": audio_eval_result,
    }


def release_phase_resources(
    phase_name: str,
    *,
    backend: str,
    model_name: str,
    clear_bert_cache: bool = False,
    clear_clip_cache: bool = False,
) -> None:
    print(f"[cleanup] Releasing resources after {phase_name} phase")

    if backend == "vllm":
        release_vllm_models(model_name=model_name)

    if clear_bert_cache:
        clear_bert_score_cache()
    if clear_clip_cache:
        clear_clip_model_cache()


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
    text_candidate_counts = Counter(
        result.get("candidate", {}).get("text", {}).get("status", "unknown")
        for result in results
    )
    image_candidate_counts = Counter(
        result.get("candidate", {}).get("image", {}).get("status", "unknown")
        for result in results
    )
    speech_candidate_counts = Counter(
        result.get("candidate", {}).get("speech", {}).get("status", "unknown")
        for result in results
    )
    text_eval_status_counts = Counter(
        result.get("text", {}).get("status", "unknown")
        for result in results
    )
    image_eval_status_counts = Counter(
        result.get("image", {}).get("status", "unknown")
        for result in results
    )
    speech_eval_status_counts = Counter(
        result.get("speech", {}).get("status", "unknown")
        for result in results
    )
    speech_parse_method_counts = Counter(
        result.get("speech", {})
        .get("audio_flamingo_next_eval", {})
        .get("_parse_metadata", {})
        .get("parsing_method", "not_available")
        for result in results
        if result.get("speech", {}).get("audio_flamingo_next_eval") is not None
    )

    return {
        "dataset_jsonl_path": str(args.dataset_jsonl_path),
        "results_root": str(args.results_root),
        "output_path": str(args.output_path),
        "summary_output_path": str(args.summary_output_path),
        "text_model_name": args.text_model_name,
        "image_model_name": args.image_model_name,
        "speech_model_name": args.speech_model_name,
        "text_backend": args.text_backend,
        "image_backend": args.image_backend,
        "json_repair_backend": args.json_repair_backend,
        "text_batch_size": args.text_batch_size,
        "image_batch_size": args.image_batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "total_dataset_records": total_dataset_records,
        "evaluated_record_count": len(results),
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "record_status_counts": dict(record_status_counts),
        "candidate_status_counts": {
            "text": dict(text_candidate_counts),
            "image": dict(image_candidate_counts),
            "speech": dict(speech_candidate_counts),
        },
        "evaluation_status_counts": {
            "text": dict(text_eval_status_counts),
            "image": dict(image_eval_status_counts),
            "speech": dict(speech_eval_status_counts),
        },
        "speech_parse_method_counts": dict(speech_parse_method_counts),
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


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset_jsonl_path)
    text_vllm_kwargs = build_text_vllm_kwargs(args)
    image_vllm_kwargs = build_image_vllm_kwargs(args)

    start_index = max(args.start_index, 0)
    end_index = len(records) if args.end_index is None else min(args.end_index, len(records))
    if start_index >= end_index:
        raise ValueError(
            f"Invalid evaluation range: start_index={start_index}, end_index={end_index}"
        )

    total_selected_records = end_index - start_index
    ordered_slots: List[Dict[str, Any]] = []

    print(
        f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"text_backend={args.text_backend} image_backend={args.image_backend} "
        f"json_repair_backend={args.json_repair_backend}"
    )

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
            ordered_slots.append({"kind": "entry", "entry": entry})
        except Exception as error:
            print(
                f"[prepare failure] dataset_index={dataset_index} "
                f"{type(error).__name__}: {error}"
            )
            ordered_slots.append(
                {
                    "kind": "error",
                    "dataset_index": dataset_index,
                    "result": build_error_result(dataset_index, record, error),
                }
            )

    valid_entries = [slot["entry"] for slot in ordered_slots if slot["kind"] == "entry"]

    text_results = evaluate_text_entries(
        valid_entries,
        args.text_model_name,
        backend=args.text_backend,
        vllm_kwargs=text_vllm_kwargs,
        batch_size=args.text_batch_size,
    )
    release_phase_resources(
        "text",
        backend=args.text_backend,
        model_name=args.text_model_name,
        clear_bert_cache=True,
    )

    image_results = evaluate_image_entries(
        valid_entries,
        args.image_model_name,
        backend=args.image_backend,
        vllm_kwargs=image_vllm_kwargs,
        batch_size=args.image_batch_size,
    )
    release_phase_resources(
        "image",
        backend=args.image_backend,
        model_name=args.image_model_name,
        clear_clip_cache=True,
    )

    all_results: List[Dict[str, Any]] = []
    for finalized_index, slot in enumerate(ordered_slots, start=1):
        if slot["kind"] == "error":
            dataset_index = slot["dataset_index"]
            result = slot["result"]
            print(
                f"[finalize {finalized_index}/{total_selected_records}] "
                f"dataset_index={dataset_index} preparation_error"
            )
        else:
            entry = slot["entry"]
            dataset_index = entry["dataset_index"]
            record = entry["record"]
            print(
                f"[finalize {finalized_index}/{total_selected_records}] "
                f"dataset_index={dataset_index} speech evaluation"
            )

            result = build_base_result(entry)
            result["text"] = text_results[dataset_index]
            result["image"] = image_results[dataset_index]
            result["speech"] = evaluate_speech_modality(
                args.speech_model_name,
                record,
                entry["candidate"]["speech"],
                json_repair_backend=args.json_repair_backend,
                vllm_kwargs=text_vllm_kwargs,
            )
            result["record_status"] = finalize_record_status(result)

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
                total_dataset_records=len(records),
                start_index=start_index,
                end_index=end_index,
                checkpoint_label=checkpoint_label,
            )
            print(
                f"[checkpoint] Saved intermediate results at dataset_index={dataset_index} "
                f"to {args.output_path}"
            )

    save_results_snapshot(
        all_results,
        args,
        total_dataset_records=len(records),
        start_index=start_index,
        end_index=end_index,
    )

    if args.json_repair_backend == "vllm":
        release_phase_resources(
            "speech-json-repair",
            backend=args.json_repair_backend,
            model_name=args.text_model_name,
        )


if __name__ == "__main__":
    main()
