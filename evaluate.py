import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_metrics import extract_behaviors_from_text as extract_behaviors_legacy
from eval_metrics_improved import extract_behaviors_from_text as extract_behaviors_improved
from metrics import compute_final_score
from solidity_compiler import quick_validate


DEFAULT_GENERATED_DIR = r"D:\AgentFSM\baseline\results\generated"
DEFAULT_OUTPUT_CSV = r"D:\AgentFSM\baseline\results\evaluation_only_metrics.csv"
DEFAULT_COMPARISON_CSV = r"D:\AgentFSM\baseline\results\evaluation_method_comparison.csv"
DEFAULT_CONTRACTS_JSONL = r"D:\AgentFSM\preprocess\preprocess\outputs\contracts_preprocessed.jsonl"
DEFAULT_RANKING_METRIC = "BehavioralAlignmentIndex"
DEFAULT_EXTRACTOR = "improved"
AGENT_OUTPUT_ROOT = r"D:\AgentFSM\Outputs"


def sanitize_code_for_baseline(code_text: str) -> str:
    """Remove injected semantic noise and empty shells before behavior extraction."""
    text = str(code_text or "")

    # Remove known AgentFSM semantic injection blocks.
    text = re.sub(
        r"\n?//\s*\[AgentFSM Semantic Injection[^\n]*\n\s*/\*.*?\*/\s*",
        "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove block comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Remove line comments but keep SPDX line if present.
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("// SPDX"):
            cleaned_lines.append(line)
            continue
        line = re.sub(r"//.*", "", line)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Remove empty constructors/functions/modifiers.
    for _ in range(8):
        prev = text
        text = re.sub(r"\bconstructor\s*\([^)]*\)\s*(?:public|external|internal|private)?\s*\{\s*\}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfunction\s+\w+\s*\([^)]*\)\s*(?:public|external|internal|private|view|pure|payable|virtual|override|returns\s*\([^)]*\)|\s)*\{\s*\}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmodifier\s+\w+\s*\([^)]*\)?\s*\{\s*\}", "", text, flags=re.IGNORECASE)
        if text == prev:
            break

    # Collapse excessive blank lines to stabilize extractor behavior.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# One-switch presets for common model output directories.
# Use --profile to select one without repeatedly passing long paths.
EVAL_PRESETS: Dict[str, Dict[str, str]] = {
    "deepseek-chat": {
        "generated_dir": r"D:\AgentFSM\baseline\results\generated\deepseek-chat",
        "output_csv": r"D:\AgentFSM\baseline\results\deepseek_chat_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\deepseek_chat_method_comparison.csv",
        "models": "deepseek-chat",
    },
    "gpt-4o-mini": {
        "generated_dir": r"D:\AgentFSM\baseline\results\generated\gpt-4o-mini",
        "output_csv": r"D:\AgentFSM\baseline\results\gpt_4o_mini_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\gpt_4o_mini_method_comparison.csv",
        "models": "gpt-4o-mini",
    },
    "gemini-2.5-flash-lite": {
        "generated_dir": r"D:\AgentFSM\baseline\results\generated\gemini-2.5-flash-lite",
        "output_csv": r"D:\AgentFSM\baseline\results\gemini_2_5_flash_lite_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\gemini_2_5_flash_lite_method_comparison.csv",
        "models": "gemini-2.5-flash-lite",
    },
    "agent2-deepseek-chat": {
        "generated_dir": os.path.join(AGENT_OUTPUT_ROOT, "agent2_code_deepseek_chat"),
        "output_csv": r"D:\AgentFSM\baseline\results\agent2_code_deepseek_chat_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\agent2_code_deepseek_chat_method_comparison.csv",
        "models": "deepseek-chat",
    },
    "agent2-gpt-4o-mini": {
        "generated_dir": os.path.join(AGENT_OUTPUT_ROOT, "agent2_code_gpt_4o_mini"),
        "output_csv": r"D:\AgentFSM\baseline\results\agent2_code_gpt_4o_mini_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\agent2_code_gpt_4o_mini_method_comparison.csv",
        "models": "gpt-4o-mini",
    },
    "agent2-gemini-2.5-flash-lite": {
        "generated_dir": os.path.join(AGENT_OUTPUT_ROOT, "agent2_code_gemini_2_5_flash_lite"),
        "output_csv": r"D:\AgentFSM\baseline\results\agent2_code_gemini_2_5_flash_lite_metrics.csv",
        "comparison_csv": r"D:\AgentFSM\baseline\results\agent2_code_gemini_2_5_flash_lite_method_comparison.csv",
        "models": "gemini-2.5-flash-lite",
    },
}

# Default run order when no explicit generated_dir/model args are passed.
DEFAULT_RUN_PRESETS: List[str] = [
    "deepseek-chat",
    "gpt-4o-mini",
    "gemini-2.5-flash-lite",
    "agent2-deepseek-chat",
    "agent2-gpt-4o-mini",
    "agent2-gemini-2.5-flash-lite",
]

KNOWN_METHODS = [
    "bpmn",
    "cml",
    "spesc",
    "symboleo",
    "prompt_only",
]


def _resolve_extractor(name: str):
    extractor = str(name or DEFAULT_EXTRACTOR).strip().lower()
    if extractor == "legacy":
        return extract_behaviors_legacy
    return extract_behaviors_improved


def compute_fsm_structure_score(code: str) -> float:
    """Evaluate structural architecture integrity of FSM (0.0 to 1.0)"""
    has_enum = bool(re.search(r'\benum\s+', code))
    has_state_guard = bool(re.search(r'require\s*\([^)]*(?:state|status)|if\s*\([^)]*(?:state|status)', code, re.IGNORECASE))
    has_access_guard = bool(re.search(r'require\s*\([^)]*msg\.sender|\bmodifier\s+', code, re.IGNORECASE))
    has_state_transition = bool(re.search(r'\b(?:state|status)\s*=\s*', code, re.IGNORECASE))
    
    modifiers_count = len(re.findall(r'\bmodifier\s+\w+', code))
    requires_count = len(re.findall(r'\brequire\s*\(', code))
    functions_count = len(re.findall(r'\bfunction\s+\w+', code))
    
    security_density = (modifiers_count + requires_count) / max(functions_count, 1)
    
    score = (
        (0.25 if has_enum else 0.0) +
        (0.25 if has_state_guard else 0.0) +
        (0.25 if has_state_transition else 0.0) +
        (0.15 if has_access_guard else 0.0) +
        (0.10 * min(security_density, 1.0))
    )
    return round(score, 4)


def compute_redundancy_penalty(code: str, code_behaviors: List[Dict[str, Any]], legal_text: str = "") -> float:
    """Estimate boilerplate / repetition / recitation pressure.

    Lower values mean the output is more likely to be verbose, repetitive, or
    a plain recitation of contract text rather than an implementation.
    """
    code_lines = [line.strip() for line in str(code or "").splitlines() if line.strip()]
    if not code_lines:
        return 0.0

    def _normalize_line(line: str) -> str:
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r"/\*.*?\*/", "", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip().lower()

    normalized_lines = [_normalize_line(line) for line in code_lines]
    normalized_lines = [line for line in normalized_lines if line]
    if not normalized_lines:
        return 0.0

    unique_line_ratio = len(set(normalized_lines)) / len(normalized_lines)
    duplicate_line_pressure = 1.0 - unique_line_ratio

    textual_lines = 0
    for line in code_lines:
        alpha_count = len(re.findall(r"[A-Za-z]", line))
        symbol_count = len(re.findall(r"[{}();:=<>\[\]+\-*/]", line))
        if alpha_count >= 20 and alpha_count > symbol_count * 3:
            textual_lines += 1
    recitation_pressure = textual_lines / len(code_lines)

    actions = [str(b.get("action", "")).strip().lower() for b in code_behaviors if isinstance(b, dict)]
    actions = [a for a in actions if a]
    if actions:
        unique_action_ratio = len(set(actions)) / len(actions)
        repeated_action_pressure = 1.0 - unique_action_ratio
    else:
        repeated_action_pressure = 0.0

    legal_pressure = 0.0
    if legal_text:
        legal_words = set(re.findall(r"[a-z]{4,}", legal_text.lower()))
        code_words = set(re.findall(r"[a-z]{4,}", str(code or "").lower()))
        if legal_words and code_words:
            legal_pressure = len(legal_words & code_words) / len(legal_words | code_words)

    penalty = 1.0
    penalty -= 0.25 * duplicate_line_pressure
    penalty -= 0.25 * recitation_pressure
    penalty -= 0.30 * repeated_action_pressure
    penalty -= 0.20 * legal_pressure
    return round(max(0.45, min(1.0, penalty)), 4)

def _extract_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        full = value.get("full")
        if isinstance(full, str) and full.strip():
            return full

        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text

        chunks = value.get("chunks")
        if isinstance(chunks, (list, tuple)):
            parts: List[str] = []
            for item in chunks:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text", item.get("chunk", ""))
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
            return "\n".join(parts)

        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (list, tuple)):
        parts = [_extract_text(v) for v in value]
        return "\n".join(p for p in parts if p.strip())

    return str(value)


def _extract_code_text(data: Dict[str, Any]) -> str:
    for key in (
        "solidity_code",
        "solidity_code_or_error",
        "generated_code",
        "code",
        "output",
    ):
        if key in data and data.get(key) is not None:
            return _extract_text(data.get(key)).strip()
    return ""


def _looks_like_error_output(text: str) -> bool:
    s = str(text or "").strip().lower()
    return s.startswith("error") or s.startswith("traceback")


def _infer_method_from_filename(file_name: str) -> str:
    lower = file_name.lower()
    for method in KNOWN_METHODS:
        if f"_{method}_" in lower or lower.endswith(f"_{method}.json"):
            return method

    if "secure_iter" in lower:
        return "agent2"

    return "unknown"


def _infer_model_from_path(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    lower = file_name.lower()
    m = re.search(r"_(bpmn|cml|spesc|symboleo|prompt_only|agent2)_(.+)\\.json$", lower)
    if m:
        return m.group(2)

    full_lower = file_path.lower().replace("\\", "/")
    m2 = re.search(r"agent2_code_([a-z0-9_.-]+)", full_lower)
    if m2:
        model_name = m2.group(1).replace("_", "-")
        if model_name == "gemini-2-5-flash-lite":
            return "gemini-2.5-flash-lite"
        return model_name

    if "deepseek" in lower:
        return "deepseek-chat"
    if "gpt-4o-mini" in lower:
        return "gpt-4o-mini"

    return "unknown"


def _infer_contract_id_from_filename(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    stem = re.sub(r"_(secure_iter\d+|working)$", "", stem, flags=re.IGNORECASE)

    if stem.startswith("CUAD_"):
        return f"CUAD::{stem[len('CUAD_'):]}"
    if stem.startswith("MAUD_"):
        return f"MAUD::{stem[len('MAUD_'):]}"

    return ""


def _collect_result_files(generated_dir: str, extra_files: Optional[Sequence[str]]) -> List[str]:
    collected: List[str] = []

    if generated_dir and os.path.isdir(generated_dir):
        for file_name in sorted(os.listdir(generated_dir)):
            if not file_name.lower().endswith((".json", ".sol")):
                continue
            if file_name.lower().endswith(".sol") and not file_name.lower().endswith("_working.sol"):
                continue
            if file_name.lower().endswith(".json") and "agent2_code" in generated_dir.lower():
                continue
            collected.append(os.path.abspath(os.path.join(generated_dir, file_name)))

    for path in extra_files or []:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if not abs_path.lower().endswith((".json", ".sol")):
            continue
        if os.path.isfile(abs_path):
            collected.append(abs_path)

    uniq: List[str] = []
    seen = set()
    for path in collected:
        if path in seen:
            continue
        seen.add(path)
        uniq.append(path)

    return uniq


def _canonical_contract_candidates(contract_id: str) -> List[str]:
    cid = str(contract_id or "").strip()
    if not cid:
        return []

    out = [cid, cid.lower()]

    if "::" in cid:
        head, tail = cid.split("::", 1)
        out.extend([
            tail,
            tail.lower(),
            f"{head}__{tail}",
            f"{head.lower()}__{tail.lower()}",
        ])

    if "__" in cid:
        head, tail = cid.split("__", 1)
        out.extend([
            f"{head}::{tail}",
            f"{head.lower()}::{tail}",
            tail,
            tail.lower(),
        ])

    uniq: List[str] = []
    seen = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def load_contracts_map(jsonl_path: str) -> Dict[str, str]:
    contract_map: Dict[str, str] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = _extract_text(obj.get("text", "")).strip()
            if not text:
                continue

            uid = str(obj.get("uid", "")).strip()
            contract_id = str(obj.get("contract_id", "")).strip()

            keys: Set[str] = set()
            if uid:
                keys.update(_canonical_contract_candidates(uid))
            if contract_id:
                keys.update(_canonical_contract_candidates(contract_id))

            for key in keys:
                contract_map[key] = text

    print(f"[INFO] Loaded {len(contract_map)} contract keys from jsonl")
    return contract_map


def _resolve_legal_text(contract_id: str, contract_map: Dict[str, str]) -> str:
    for key in _canonical_contract_candidates(contract_id):
        if key in contract_map:
            return contract_map[key]
    return ""


def _parse_contract_filter(raw: str) -> Set[str]:
    if not raw:
        return set()

    items = [x.strip() for x in raw.split(",") if x.strip()]
    out: Set[str] = set()
    for item in items:
        out.update(_canonical_contract_candidates(item))
    return out


def _parse_label_filter(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _run_default_presets(
    contracts_jsonl: str,
    ranking_metric: str,
    contract_filter: Optional[Set[str]] = None,
    method_filter: Optional[Set[str]] = None,
    model_filter: Optional[Set[str]] = None,
    extractor_name: str = DEFAULT_EXTRACTOR,
    apply_bloat_penalty: bool = True,
    include_structure_score: bool = True,
    sanitize_baseline_code: bool = False,
    batch_size: int = 0,
    batch_index: int = 0,
) -> None:
    """Run evaluation for all default presets using paths configured in code."""
    for preset_name in DEFAULT_RUN_PRESETS:
        cfg = EVAL_PRESETS.get(preset_name)
        if not cfg:
            print(f"[WARNING] Skip missing preset config: {preset_name}")
            continue

        generated_dir = cfg["generated_dir"]
        output_csv = cfg["output_csv"]
        comparison_csv = cfg["comparison_csv"]
        preset_models = _parse_label_filter(cfg.get("models", ""))

        # If caller provided model_filter, intersect with preset model scope.
        effective_model_filter = preset_models
        if model_filter:
            effective_model_filter = preset_models.intersection(model_filter) if preset_models else model_filter
        if model_filter and not effective_model_filter:
            print(f"[INFO] Skip preset {preset_name}: filtered out by --models")
            continue

        print(f"[INFO] Running preset: {preset_name}")
        evaluate_from_generated(
            generated_dir=generated_dir,
            contracts_jsonl=contracts_jsonl,
            output_csv=output_csv,
            comparison_csv=comparison_csv,
            contract_filter=contract_filter,
            method_filter=method_filter,
            model_filter=effective_model_filter,
            extra_generated_files=None,
            ranking_metric=ranking_metric,
            extractor_name=extractor_name,
            apply_bloat_penalty=apply_bloat_penalty,
            include_structure_score=include_structure_score,
            sanitize_baseline_code=sanitize_baseline_code,
            batch_size=batch_size,
            batch_index=batch_index,
        )


def _contract_in_filter(contract_id: str, contract_filter: Set[str]) -> bool:
    if not contract_filter:
        return True
    candidates = _canonical_contract_candidates(contract_id)
    return any(c in contract_filter for c in candidates)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _build_comparison_rows(rows: List[Dict[str, Any]], ranking_metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("contract_id", ""), row.get("model", ""))].append(row)

    comparison_rows: List[Dict[str, Any]] = []

    for (contract_id, model), group in grouped.items():
        ranked = sorted(
            group,
            key=lambda x: (
                _as_float(x.get(ranking_metric, 0.0)),
                _as_float(x.get("BehavioralAlignmentIndex", 0.0)),
                _as_float(x.get("FinalScore", 0.0)),
            ),
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        ranking_str = " > ".join(
            f"{item.get('method', 'unknown')}:{_as_float(item.get(ranking_metric, 0.0)):.4f}"
            for item in ranked
        )

        comparison_rows.append(
            {
                "contract_id": contract_id,
                "model": model,
                "methods_evaluated": len(ranked),
                "ranking_metric": ranking_metric,
                "best_method": best.get("method", "unknown"),
                "best_score": round(_as_float(best.get(ranking_metric, 0.0)), 4),
                "second_method": second.get("method", "") if second else "",
                "second_score": round(_as_float(second.get(ranking_metric, 0.0)), 4) if second else "",
                "margin_best_second": round(
                    _as_float(best.get(ranking_metric, 0.0)) - _as_float(second.get(ranking_metric, 0.0)),
                    4,
                ) if second else "",
                "ranking": ranking_str,
            }
        )

    comparison_rows.sort(key=lambda x: (x["contract_id"], x["model"]))
    return comparison_rows


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_from_generated(
    generated_dir: str,
    contracts_jsonl: str,
    output_csv: str,
    comparison_csv: str,
    contract_filter: Optional[Set[str]] = None,
    method_filter: Optional[Set[str]] = None,
    model_filter: Optional[Set[str]] = None,
    extra_generated_files: Optional[Sequence[str]] = None,
    ranking_metric: str = "BehavioralAlignmentIndex",
    extractor_name: str = DEFAULT_EXTRACTOR,
    apply_bloat_penalty: bool = True,
    include_structure_score: bool = True,
    sanitize_baseline_code: bool = False,
    batch_size: int = 0,
    batch_index: int = 0,
) -> None:
    extract_behaviors = _resolve_extractor(extractor_name)
    contract_map = load_contracts_map(contracts_jsonl)
    files = _collect_result_files(generated_dir, extra_generated_files)
    legal_behaviors_cache: Dict[str, List[Dict[str, Any]]] = {}
    # COMPILE_WEIGHT blends compilability into FinalScore: 0.0 -> ignore, 1.0 -> only compilability
    try:
        compile_weight = float(os.getenv("COMPILE_WEIGHT", "0.0"))
    except Exception:
        compile_weight = 0.0

    total_files = len(files)
    if batch_size and batch_size > 0:
        start = max(batch_index, 0) * batch_size
        end = min(start + batch_size, total_files)
        files = files[start:end]
        print(f"[INFO] Found {total_files} result files, evaluating batch {batch_index} ({start}:{end})")
    else:
        print(f"[INFO] Found {total_files} result files")

    rows: List[Dict[str, Any]] = []

    for path in files:
        file_name = os.path.basename(path)
        ext = os.path.splitext(file_name)[1].lower()

        contract_id = ""
        method = ""
        model = ""
        code_text = ""

        if ext == ".sol":
            contract_id = _infer_contract_id_from_filename(file_name)
            if not contract_id:
                print(f"[WARNING] Skip {file_name}: cannot infer contract_id from filename")
                continue

            method = "agent2"
            model = _infer_model_from_path(path)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    code_text = f.read().strip()
            except Exception as e:
                print(f"[WARNING] Failed to read {file_name}: {e}")
                continue

        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to read {file_name}: {e}")
                continue

            contract_id = str(data.get("contract_id") or data.get("fsm_id") or "").strip()
            if not contract_id:
                print(f"[WARNING] Skip {file_name}: missing contract_id")
                continue

            method = str(data.get("method") or _infer_method_from_filename(file_name)).strip()
            model = str(data.get("model") or _infer_model_from_path(path)).strip()
            code_text = _extract_code_text(data).strip()

        if not contract_id:
            print(f"[WARNING] Skip {file_name}: missing contract_id")
            continue

        if not _contract_in_filter(contract_id, contract_filter or set()):
            continue

        if method_filter and method.lower() not in method_filter:
            continue
        if model_filter and model.lower() not in model_filter:
            continue

        legal_text = _resolve_legal_text(contract_id, contract_map)
        if not legal_text:
            print(f"[WARNING] Contract text not found for {contract_id} ({file_name})")
            continue

        if not code_text or _looks_like_error_output(code_text):
            print(f"[WARNING] Skip invalid output: {contract_id}-{method}-{model}")
            continue

        if sanitize_baseline_code:
            code_text = sanitize_code_for_baseline(code_text)
            if not code_text:
                print(f"[WARNING] Skip empty code after sanitize: {contract_id}-{method}-{model}")
                continue

        cache_key = contract_id.strip().lower()
        if cache_key in legal_behaviors_cache:
            legal_behaviors = legal_behaviors_cache[cache_key]
        else:
            legal_behaviors = extract_behaviors(legal_text, "l")
            legal_behaviors_cache[cache_key] = legal_behaviors
        code_behaviors = extract_behaviors(code_text, "s")

        try:
            compile_success = bool(quick_validate(code_text, contract_id)) if code_text else False
        except Exception:
            compile_success = False

        try:
            metrics = compute_final_score(
                legal_behaviors,
                code_behaviors,
                alpha=0.2, beta=0.45, gamma=0.2, delta=0.15,
                precision_threshold=0.7
            )
        except Exception as e:
            print(f"[ERROR] Metric failed for {contract_id}-{method}-{model}: {e}")
            continue

        rows.append(
            {
                "contract_id": contract_id,
                "method": method,
                "model": model,
                "source_file": file_name,
                "source_format": "sol" if ext == ".sol" else "json",
                "Fitness": round(_as_float(metrics.get("Fitness", 0.0)), 4),
                "Precision": round(_as_float(metrics.get("Precision", 0.0)), 4),
                "TemporalConsistency": round(_as_float(metrics.get("TemporalConsistency", 0.0)), 4),
                "TemporalPrecision": round(_as_float(metrics.get("TemporalPrecision", 0.0)), 4),
                "TemporalF1": round(_as_float(metrics.get("TemporalF1", 0.0)), 4),
                "DeonticSatisfaction": round(_as_float(metrics.get("DeonticSatisfaction", 0.0)), 4),
                "ConditionAlignment": round(_as_float(metrics.get("ConditionAlignment", 0.0)), 4),
                "BehavioralAlignmentIndex": round(
                    _as_float(metrics.get("BehavioralAlignmentIndex", metrics.get("FinalScore", 0.0))),
                    4,
                ),
                "FinalScore": round(_as_float(metrics.get("FinalScore", 0.0)), 4),
                "legal_behavior_count": len(legal_behaviors),
                "code_behavior_count": len(code_behaviors),
            }
        )
            
        entry = rows.pop()  # Remove the appended dictionary and modify it safely

        if apply_bloat_penalty:
            code_bloat_penalty = 1.0
            if len(code_behaviors) > len(legal_behaviors) and len(legal_behaviors) > 0:
                # Drop score by how much they overgenerated (max penalty to 30%)
                code_bloat_penalty = max(0.3, len(legal_behaviors) / len(code_behaviors))

            # Penalize NLP precision if hallucinated
            entry["Precision"] = round(entry["Precision"] * code_bloat_penalty, 4)
            entry["BehavioralAlignmentIndex"] = round(entry["BehavioralAlignmentIndex"] * code_bloat_penalty, 4)

        redundancy_penalty = compute_redundancy_penalty(code_text, code_behaviors, legal_text)

        if include_structure_score:
            fsm_struct_score = compute_fsm_structure_score(code_text)
            if compile_success:
                fsm_struct_score = min(1.0, round(fsm_struct_score + 0.12, 4))
            entry["FSM_StructureScore"] = fsm_struct_score
            # Combine behavior with structural quality, then apply redundancy pressure.
            entry["ComprehensiveFSMQuality"] = round(
                ((entry["BehavioralAlignmentIndex"] * 0.55) + (fsm_struct_score * 0.45)) * redundancy_penalty,
                4,
            )
        else:
            entry["FSM_StructureScore"] = ""
            entry["ComprehensiveFSMQuality"] = ""

        # Redundancy / recitation pressure should hurt precision-oriented scores the most.
        entry["Precision"] = round(entry["Precision"] * redundancy_penalty, 4)
        entry["BehavioralAlignmentIndex"] = round(entry["BehavioralAlignmentIndex"] * redundancy_penalty, 4)

        rows.append(entry)

        fs_val = float(entry.get("FinalScore", 0.0))
        adjusted = (1.0 - compile_weight) * fs_val + compile_weight * (1.0 if compile_success else 0.0)
        entry["compile_success"] = compile_success
        entry["adjusted_final_score"] = round(adjusted, 4)
    if not rows:
        print("[WARNING] No valid evaluation rows.")
        return

    metric_fieldnames = [
        "contract_id",
        "method",
        "model",
        "source_file",
        "source_format",
        "Fitness",
        "Precision",
        "TemporalConsistency",
        "TemporalPrecision",
        "TemporalF1",
        "DeonticSatisfaction",
        "ConditionAlignment",
        "BehavioralAlignmentIndex",
        "FinalScore",
        "legal_behavior_count",
        "code_behavior_count",
        "FSM_StructureScore",
        "ComprehensiveFSMQuality",
        "compile_success",
        "adjusted_final_score",
    ]

    rows.sort(key=lambda x: (x["contract_id"], x["model"], x["method"]))
    _write_csv(output_csv, rows, metric_fieldnames)
    print(f"[SUCCESS] Evaluation completed -> {output_csv}")

    comparison_rows = _build_comparison_rows(rows, ranking_metric)
    comparison_fieldnames = [
        "contract_id",
        "model",
        "methods_evaluated",
        "ranking_metric",
        "best_method",
        "best_score",
        "second_method",
        "second_score",
        "margin_best_second",
        "ranking",
    ]
    _write_csv(comparison_csv, comparison_rows, comparison_fieldnames)
    print(f"[SUCCESS] Method comparison saved -> {comparison_csv}")

    print("[INFO] Top method per contract/model:")
    for row in comparison_rows:
        print(
            f"  - {row['contract_id']} [{row['model']}] => {row['best_method']} "
            f"({row['ranking_metric']}={row['best_score']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--contracts_jsonl", type=str, default=DEFAULT_CONTRACTS_JSONL)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--comparison_csv", type=str, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default=DEFAULT_RANKING_METRIC,
        choices=["BehavioralAlignmentIndex", "FinalScore"],
    )
    parser.add_argument(
        "--contract_ids",
        type=str,
        default="",
        help="Comma-separated contract ids to evaluate (optional)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method filter (e.g. bpmn,cml,spesc,symboleo,agent2)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model filter (e.g. deepseek-chat,unknown)",
    )
    parser.add_argument(
        "--extra_generated_file",
        action="append",
        default=[],
        help="Extra generated JSON file path, repeatable",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=DEFAULT_EXTRACTOR,
        choices=["improved", "legacy"],
        help="Behavior extractor backend. Use legacy to align with eval_metrics-based scripts.",
    )
    parser.add_argument(
        "--disable_bloat_penalty",
        action="store_true",
        help="Disable post-score bloat penalty to keep raw compute_final_score Precision/BAI.",
    )
    parser.add_argument(
        "--disable_structure_score",
        action="store_true",
        help="Disable FSM structure score and ComprehensiveFSMQuality columns.",
    )
    parser.add_argument(
        "--sanitize_baseline_code",
        action="store_true",
        help="Remove semantic injections/comments/empty shells before behavior extraction.",
    )
    parser.add_argument(
        "--strict_baseline_full",
        action="store_true",
        help="Match the baseline strict CSV framework: legacy extractor, no bloat penalty, no FSM structure score.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Evaluate files in batches. 0 means evaluate all files.",
    )
    parser.add_argument(
        "--batch_index",
        type=int,
        default=0,
        help="0-based batch index when --batch_size is set.",
    )

    args = parser.parse_args()

    if args.strict_baseline_full:
        args.extractor = "legacy"
        args.disable_bloat_penalty = True
        args.disable_structure_score = True

    contract_filter = _parse_contract_filter(args.contract_ids)
    method_filter = _parse_label_filter(args.methods)
    model_filter = _parse_label_filter(args.models)

    # If user keeps default generated_dir and no extra file, run all configured presets.
    should_run_presets = (
        args.generated_dir == DEFAULT_GENERATED_DIR
        and not args.extra_generated_file
    )

    if should_run_presets:
        _run_default_presets(
            contracts_jsonl=args.contracts_jsonl,
            ranking_metric=args.ranking_metric,
            contract_filter=contract_filter,
            method_filter=method_filter,
            model_filter=model_filter,
            extractor_name=args.extractor,
            apply_bloat_penalty=not args.disable_bloat_penalty,
            include_structure_score=not args.disable_structure_score,
            sanitize_baseline_code=args.sanitize_baseline_code,
            batch_size=args.batch_size,
            batch_index=args.batch_index,
        )
    else:
        evaluate_from_generated(
            generated_dir=args.generated_dir,
            contracts_jsonl=args.contracts_jsonl,
            output_csv=args.output_csv,
            comparison_csv=args.comparison_csv,
            contract_filter=contract_filter,
            method_filter=method_filter,
            model_filter=model_filter,
            extra_generated_files=args.extra_generated_file,
            ranking_metric=args.ranking_metric,
            extractor_name=args.extractor,
            apply_bloat_penalty=not args.disable_bloat_penalty,
            include_structure_score=not args.disable_structure_score,
            sanitize_baseline_code=args.sanitize_baseline_code,
            batch_size=args.batch_size,
            batch_index=args.batch_index,
        )
