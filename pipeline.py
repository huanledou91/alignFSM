import json
import logging
import urllib3
import time
import re
import os
import tiktoken
from tqdm import tqdm
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from agent1_contractFSM import Agent1FSM
from agent2_codegen import Agent2FSMToContract, sanitize_filename
from agent3_codeFSM import Agent3ContractToFSM
from agent4_alignFSM import Agent4FSMAnalyzer

# ===================== 基础配置 =====================
DATASET_PATH = r"D:\AgentFSM\preprocess\preprocess\outputs\contracts_preprocessed.jsonl"
MODEL_NAMES = ["gpt-4o-mini"]
MAX_ITER = 3
RUN_ONE_SAMPLE = False  # True: 只跑一个样本
SAMPLE_COUNT = 19
SAMPLE_START_INDEX = 81 # 0-based; 1 means the second sample
AGENT2_BASE_OUTPUT_DIR = "./Outputs/agent2_code"
AGENT2_INCLUDE_SECURITY_RULES = False
AGENT2_STRICT_SEMANTIC_CHECKS = False
AGENT1_CACHE_BASE_DIR = "./Outputs/agent1_fsm_cache"
AGENT1_CACHE_READ = True
AGENT1_CACHE_WRITE = True
# 合同级并行配置：当合同数大于阈值时自动并行处理
ENABLE_CONTRACT_PARALLEL = True
PARALLEL_CONTRACT_THRESHOLD = 100
MAX_CONTRACT_WORKERS = int(os.getenv("PIPELINE_MAX_CONTRACT_WORKERS", "4"))

# Token统计配置（tiktoken模型映射）
TOKEN_ENCODING_MAP = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    # 可扩展其他模型
}

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MultiModelPipeline")

# 全局统计数据存储
global_stats = {
    "total_start_time": None,
    "total_end_time": None,
    "models": {},
    "summary": {
        "total_time_seconds": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "processed_contracts": 0
    }
}

# ===================== 工具函数 =====================
def calculate_tokens(text, model_name):
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(TOKEN_ENCODING_MAP.get(model_name, "gpt-4o-mini"))
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"计算token失败：{e}，返回0")
        return 0

def save_stats_to_json(file_path="./pipeline_stats.json"):
    """
    将统计数据保存为JSON文件
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"统计数据已保存至：{file_path}")
    except Exception as e:
        logger.error(f"保存统计数据失败：{e}")


def _build_agent1_cache_path(agent1_cache_dir, contract_id):
    safe_id = sanitize_filename(str(contract_id or "unknown_contract"))
    return os.path.join(agent1_cache_dir, f"{safe_id}_agent1_fsm.json")


def _load_agent1_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Agent1 cache is not a dict, ignore: {cache_path}")
            return None
        return data
    except Exception as e:
        logger.warning(f"Load Agent1 cache failed: {cache_path}, error={e}")
        return None


def _save_agent1_cache(cache_path, contract_id, model_name, legal_fsm):
    try:
        cache_payload = {
            "contract_id": contract_id,
            "model_name": model_name,
            "saved_at": int(time.time()),
            "legal_fsm": legal_fsm,
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Save Agent1 cache failed: {cache_path}, error={e}")


def _normalize_issue_text(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    # 去掉常见严重程度前缀，避免同一问题因前缀不同被视为新问题
    text = re.sub(r"^[^a-z0-9]*(critical|high|medium|low)\s*[:\-]\s*", "", text)
    return text


def _issue_fingerprint(issue):
    if isinstance(issue, dict):
        issue_type = _normalize_issue_text(issue.get("issue_type", "unknown"))
        location = _normalize_issue_text(issue.get("location", "unknown"))
        description = _normalize_issue_text(issue.get("description", ""))
        return f"{issue_type}|{location}|{description}"
    return _normalize_issue_text(issue)


def _extract_issue_type_from_text(text):
    s = str(text or "").strip()
    if not s:
        return "UNKNOWN"
    m = re.search(r"\]\s*([A-Z_]{3,})\s*:", s)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Z_]{3,})\s*:", s)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def _normalize_issue_severity(value):
    sev = str(value or "").strip().lower()
    if sev in {"critical", "high", "medium", "low"}:
        return sev
    return "medium"


def _extract_issue_severity_from_text(text):
    s = str(text or "").strip().lower()
    if not s:
        return "medium"
    if "critical" in s:
        return "critical"
    if "high" in s:
        return "high"
    if "low" in s:
        return "low"
    return "medium"


def _severity_weight(severity):
    severity_weights = {
        "critical": 5.0,
        "high": 3.0,
        "medium": 1.0,
        "low": 0.5,
    }
    return severity_weights.get(_normalize_issue_severity(severity), 1.0)


def _compute_feedback_risk_score(feedback, allowed_issue_types=None):
    """基于严重程度加权的风险分，用于更稳定的收敛判断。"""
    if not isinstance(feedback, dict):
        return 0.0, 0

    allowed_types = {str(t).strip().upper() for t in (allowed_issue_types or []) if str(t).strip()}
    risk_score = 0.0
    issue_count = 0
    seen_this_round = set()

    structured = feedback.get("structured_issues", [])
    if isinstance(structured, list) and structured:
        for issue in structured:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("issue_type", "UNKNOWN")).strip().upper()
            if allowed_types and issue_type not in allowed_types:
                continue
            fp = _issue_fingerprint(issue)
            if not fp or fp in seen_this_round:
                continue
            seen_this_round.add(fp)

            issue_count += 1
            risk_score += _severity_weight(issue.get("severity", "medium"))
        return risk_score, issue_count

    raw_issues = feedback.get("issues", [])
    if not isinstance(raw_issues, list):
        raw_issues = []

    for item in raw_issues:
        issue_type = _extract_issue_type_from_text(item)
        if allowed_types and issue_type not in allowed_types:
            continue
        fp = _issue_fingerprint(item)
        if not fp or fp in seen_this_round:
            continue
        seen_this_round.add(fp)

        issue_count += 1
        risk_score += _severity_weight(_extract_issue_severity_from_text(item))

    return risk_score, issue_count


def _collect_issue_types(feedback):
    types = set()
    if not isinstance(feedback, dict):
        return types

    for issue in feedback.get("structured_issues", []) or []:
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("issue_type", "")).strip().upper()
        if issue_type and issue_type != "UNKNOWN":
            types.add(issue_type)

    if types:
        return types

    for item in feedback.get("issues", []) or []:
        issue_type = _extract_issue_type_from_text(item)
        if issue_type and issue_type != "UNKNOWN":
            types.add(issue_type)
    return types


def _format_structured_issue_for_feedback(issue):
    severity = str(issue.get("severity", "medium")).upper()
    issue_type = str(issue.get("issue_type", "UNKNOWN"))
    description = str(issue.get("description", "")).strip()
    return f"[{severity}] {issue_type}: {description}"


def _build_actionable_feedback(
    raw_feedback,
    issue_lifecycle,
    iteration_idx,
    max_actionable_issues=12,
    allowed_issue_types=None,
    high_severity_repeat_limit=2,
):
    """问题生命周期策略：新问题必传，高优先级未解决问题有限次重复传递。"""
    if not isinstance(raw_feedback, dict):
        return {}, {
            "current_issue_count": 0,
            "new_issue_count": 0,
            "actionable_issue_count": 0,
            "repeated_actionable_count": 0,
            "resolved_issue_count": 0,
        }

    actionable_feedback = dict(raw_feedback)
    allowed_types = {str(t).strip().upper() for t in (allowed_issue_types or []) if str(t).strip()}

    normalized_issues = []
    seen_this_round = set()

    structured = raw_feedback.get("structured_issues", [])
    if isinstance(structured, list) and structured:
        for issue in structured:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("issue_type", "UNKNOWN")).strip().upper()
            if allowed_types and issue_type not in allowed_types:
                continue
            fp = _issue_fingerprint(issue)
            if not fp or fp in seen_this_round:
                continue
            seen_this_round.add(fp)
            normalized_issues.append({
                "fingerprint": fp,
                "severity": _normalize_issue_severity(issue.get("severity", "medium")),
                "issue": issue,
            })
    else:
        raw_issues = raw_feedback.get("issues", [])
        if not isinstance(raw_issues, list):
            raw_issues = []

        for item in raw_issues:
            issue_type = _extract_issue_type_from_text(item)
            if allowed_types and issue_type not in allowed_types:
                continue
            fp = _issue_fingerprint(item)
            if not fp or fp in seen_this_round:
                continue
            seen_this_round.add(fp)
            severity = _extract_issue_severity_from_text(item)
            normalized_issues.append({
                "fingerprint": fp,
                "severity": severity,
                "issue": {
                    "issue_type": issue_type,
                    "severity": severity,
                    "category": "unknown",
                    "description": str(item).strip(),
                    "location": "unknown",
                    "suggestion": "Please address this behavior alignment issue based on legal constraints.",
                    "legal_reference": "not provided",
                    "contract_reference": "not provided",
                },
            })

    normalized_issues.sort(key=lambda x: _severity_weight(x.get("severity", "medium")), reverse=True)

    current_issue_fps = {entry["fingerprint"] for entry in normalized_issues}
    resolved_issue_count = 0
    for fp, lifecycle in issue_lifecycle.items():
        if lifecycle.get("active", False) and fp not in current_issue_fps:
            lifecycle["active"] = False
            lifecycle["resolved_iteration"] = iteration_idx
            resolved_issue_count += 1

    actionable_structured = []
    new_issue_count = 0
    repeated_actionable_count = 0

    for entry in normalized_issues:
        fp = entry["fingerprint"]
        severity = entry["severity"]
        issue = entry["issue"]

        lifecycle = issue_lifecycle.get(fp)
        is_new = lifecycle is None
        if is_new:
            lifecycle = {
                "first_seen_iteration": iteration_idx,
                "last_seen_iteration": iteration_idx,
                "seen_count": 0,
                "active": True,
                "resolved_iteration": None,
                "severity": severity,
                "repeat_emits": 0,
            }
            issue_lifecycle[fp] = lifecycle

        lifecycle["active"] = True
        lifecycle["resolved_iteration"] = None
        lifecycle["last_seen_iteration"] = iteration_idx
        lifecycle["seen_count"] = lifecycle.get("seen_count", 0) + 1
        lifecycle["severity"] = severity

        should_emit = False
        if is_new:
            should_emit = True
            new_issue_count += 1
        elif severity in {"critical", "high"} and lifecycle.get("repeat_emits", 0) < high_severity_repeat_limit:
            should_emit = True
            repeated_actionable_count += 1

        if should_emit and len(actionable_structured) < max_actionable_issues:
            actionable_structured.append(issue)
            if not is_new:
                lifecycle["repeat_emits"] = lifecycle.get("repeat_emits", 0) + 1

    actionable_feedback["structured_issues"] = actionable_structured
    actionable_feedback["issues"] = [_format_structured_issue_for_feedback(i) for i in actionable_structured]

    return actionable_feedback, {
        "current_issue_count": len(normalized_issues),
        "new_issue_count": new_issue_count,
        "actionable_issue_count": len(actionable_structured),
        "repeated_actionable_count": repeated_actionable_count,
        "resolved_issue_count": resolved_issue_count,
    }


# ===================== 数据加载 =====================
def load_contracts(path, limit_one=False, start_index=0):
    start_index = max(0, int(start_index or 0))
    target_count = 1 if limit_one else max(1, int(SAMPLE_COUNT or 1))

    contracts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            if not line.strip():
                continue
            contracts.append(json.loads(line))
            if len(contracts) >= target_count:
                break

    logger.info(
        f"Loaded {len(contracts)} contract(s) from index {start_index} "
        f"(limit_one={limit_one}, sample_count={target_count})"
    )

    return contracts


def _init_contract_stats():
    return {
        "start_time": time.time(),
        "end_time": None,
        "elapsed_time_seconds": 0,
        "tokens": {
            "agent1": {"prompt": 0, "completion": 0, "total": 0},
            "agent2": {"prompt": 0, "completion": 0, "total": 0},
            "agent3": {"prompt": 0, "completion": 0, "total": 0},
            "agent4": {"prompt": 0, "completion": 0, "total": 0},
            "total": {"prompt": 0, "completion": 0, "total": 0}
        }
    }


def _finalize_contract_stats(contract_stats):
    if contract_stats["end_time"] is None:
        contract_stats["end_time"] = time.time()
    contract_stats["elapsed_time_seconds"] = contract_stats["end_time"] - contract_stats["start_time"]

    contract_stats["tokens"]["total"]["prompt"] = (
        contract_stats["tokens"]["agent1"]["prompt"] +
        contract_stats["tokens"]["agent2"]["prompt"] +
        contract_stats["tokens"]["agent3"]["prompt"] +
        contract_stats["tokens"]["agent4"]["prompt"]
    )
    contract_stats["tokens"]["total"]["completion"] = (
        contract_stats["tokens"]["agent1"]["completion"] +
        contract_stats["tokens"]["agent2"]["completion"] +
        contract_stats["tokens"]["agent3"]["completion"] +
        contract_stats["tokens"]["agent4"]["completion"]
    )
    contract_stats["tokens"]["total"]["total"] = (
        contract_stats["tokens"]["total"]["prompt"] + contract_stats["tokens"]["total"]["completion"]
    )


def _create_agents(model_name, agent2_output_dir):
    agent1 = Agent1FSM(model_name=model_name)
    agent2 = Agent2FSMToContract(
        model_name=model_name,
        output_dir=agent2_output_dir,
        max_iterations=3,
        include_security_rules=AGENT2_INCLUDE_SECURITY_RULES,
        strict_semantic_checks=AGENT2_STRICT_SEMANTIC_CHECKS,
        defer_compilation_fix=True,
    )
    agent3 = Agent3ContractToFSM(model_name=model_name)
    agent4 = Agent4FSMAnalyzer(model_name=model_name)
    return agent1, agent2, agent3, agent4


def _run_single_contract(model_name, contract, agent2_output_dir, agent1_cache_dir):
    contract_id = contract.get("contract_id", "unknown_contract")
    if "::" not in contract_id and not contract_id.startswith("MAUD"):
        # some contracts don't have the explicit dataset prefix, assume CUAD if not there since we match the filenames
        expected_cid = f"CUAD::{contract_id}"
    else:
        expected_cid = contract_id
    sanitized_contract_id = sanitize_filename(expected_cid)
    
    # === SKIP LOGIC ===
    expected_out = os.path.join(agent2_output_dir, f"{sanitized_contract_id}_working.sol")
    if os.path.exists(expected_out):
        logger.info(f"[{model_name}] Skipping contract {contract_id}, output already exists.")
        return contract_id, _init_contract_stats()
    
    sanitized_contract_id = sanitize_filename(contract_id) # reset for the rest of pipeline
    
    contract_stats = _init_contract_stats()

    logger.info(f"[{model_name}] Processing contract: {contract_id}")
    agent1, agent2, agent3, agent4 = _create_agents(model_name, agent2_output_dir)
    agent1_cache_path = _build_agent1_cache_path(agent1_cache_dir, contract_id)

    try:
        # ===================== Agent1 =====================
        legal_fsm = None
        loaded_from_cache = False
        if AGENT1_CACHE_READ:
            cached = _load_agent1_cache(agent1_cache_path)
            if isinstance(cached, dict) and isinstance(cached.get("legal_fsm"), dict):
                legal_fsm = cached["legal_fsm"]
                loaded_from_cache = True
                logger.info(f"[{model_name}] Agent1 cache hit: {contract_id}")

        if legal_fsm is None:
            legal_fsm = agent1.run(contract)
            if AGENT1_CACHE_WRITE and isinstance(legal_fsm, dict):
                _save_agent1_cache(agent1_cache_path, contract_id, model_name, legal_fsm)

        if loaded_from_cache:
            contract_stats["tokens"]["agent1"]["prompt"] = 0
            contract_stats["tokens"]["agent1"]["completion"] = 0
        elif hasattr(agent1, "last_token_usage") and isinstance(agent1.last_token_usage, dict):
            contract_stats["tokens"]["agent1"]["prompt"] = agent1.last_token_usage.get("prompt_tokens", 0)
            contract_stats["tokens"]["agent1"]["completion"] = agent1.last_token_usage.get("completion_tokens", 0)
        else:
            prompt_text = json.dumps(contract, ensure_ascii=False)
            completion_text = json.dumps(legal_fsm, ensure_ascii=False)
            contract_stats["tokens"]["agent1"]["prompt"] = calculate_tokens(prompt_text, model_name)
            contract_stats["tokens"]["agent1"]["completion"] = calculate_tokens(completion_text, model_name)
        contract_stats["tokens"]["agent1"]["total"] = (
            contract_stats["tokens"]["agent1"]["prompt"] + contract_stats["tokens"]["agent1"]["completion"]
        )

        logger.info(f"[{model_name}] Agent1 返回的原始数据：{json.dumps(legal_fsm, indent=2, ensure_ascii=False)}")
        logger.info(f"[{model_name}] Agent1 返回数据的键：{list(legal_fsm.keys())}")

        legal_fsm_core = {}
        fsm_source = "none"
        if isinstance(legal_fsm, dict):
            # Accept multiple Agent1 output shapes:
            # 1) {"legal_fsm": {...}}
            # 2) {"fsm": {...}}
            # 3) {"states": [...], "transitions": [...]} at root
            if isinstance(legal_fsm.get("legal_fsm"), dict):
                legal_fsm_core = legal_fsm.get("legal_fsm", {})
                fsm_source = "legal_fsm"
            elif isinstance(legal_fsm.get("fsm"), dict):
                legal_fsm_core = legal_fsm.get("fsm", {})
                fsm_source = "fsm"
            elif isinstance(legal_fsm.get("states"), list) and isinstance(legal_fsm.get("transitions"), list):
                legal_fsm_core = legal_fsm
                fsm_source = "root"

        legal_states = legal_fsm_core.get("states", []) if isinstance(legal_fsm_core, dict) else []
        legal_transitions = legal_fsm_core.get("transitions", []) if isinstance(legal_fsm_core, dict) else []

        if (
            not isinstance(legal_fsm, dict)
            or not isinstance(legal_states, list)
            or not isinstance(legal_transitions, list)
            or len(legal_states) == 0
            or len(legal_transitions) == 0
        ):
            logger.error(
                f"[{model_name}] Agent1 failed to produce valid FSM for {contract_id}. "
                f"Expected non-empty states/transitions. source={fsm_source}, "
                f"states={len(legal_states)}, transitions={len(legal_transitions)}"
            )
            return contract_id, contract_stats

        all_constraints = []
        for candidate in (
            legal_fsm.get("all_solidity_constraints") if isinstance(legal_fsm, dict) else None,
            legal_fsm_core.get("all_solidity_constraints") if isinstance(legal_fsm_core, dict) else None,
            legal_fsm.get("solidity_constraints") if isinstance(legal_fsm, dict) else None,
            legal_fsm_core.get("solidity_constraints") if isinstance(legal_fsm_core, dict) else None,
        ):
            if isinstance(candidate, list):
                all_constraints = candidate
                break

        constraint_count = 0
        if isinstance(legal_fsm, dict) and isinstance(legal_fsm.get("constraint_count"), int):
            constraint_count = legal_fsm.get("constraint_count", 0)
        elif isinstance(legal_fsm_core, dict) and isinstance(legal_fsm_core.get("constraint_count"), int):
            constraint_count = legal_fsm_core.get("constraint_count", 0)
        else:
            constraint_count = len(all_constraints)

        fsm_payload = legal_fsm.copy() if isinstance(legal_fsm, dict) else {}
        fsm_payload["legal_fsm"] = legal_fsm_core

        fsm_result_for_agent2 = {
            "contract_id": contract.get("uid") or contract.get("contract_id", "unknown_contract"),
            "fsm": fsm_payload,
            # forward any extracted solidity constraints for Agent2
            "all_solidity_constraints": all_constraints,
            "constraint_count": constraint_count,
        }

        # ===================== 反馈闭环 =====================
        agent4_feedback = None
        final_code = None
        issue_lifecycle = {}
        previous_risk_score = None
        no_improve_rounds = 0
        best_risk_score = float("inf")
        best_code = None
        feedback_issue_cap = None
        locked_issue_types = None

        for it in range(1, MAX_ITER + 1):
            logger.info(f"[{model_name}] [{contract_id}] [Iteration {it}/{MAX_ITER}] Agent2: FSM → Code")

            # 增量修复：将上一轮代码传给Agent2，避免每轮从骨架重生成
            if isinstance(final_code, str) and final_code.strip():
                fsm_result_for_agent2["previous_contract_code"] = final_code
            else:
                fsm_result_for_agent2.pop("previous_contract_code", None)

            # ---------- Agent2 ----------
            agent2_result = agent2.run(
                fsm_result=fsm_result_for_agent2,
                agent4_feedback=agent4_feedback
            )
            # 统计Agent2的Token
            if hasattr(agent2, "last_token_usage") and isinstance(agent2.last_token_usage, dict):
                contract_stats["tokens"]["agent2"]["prompt"] += agent2.last_token_usage.get("prompt_tokens", 0)
                contract_stats["tokens"]["agent2"]["completion"] += agent2.last_token_usage.get("completion_tokens", 0)
            else:
                prompt_text = json.dumps({"fsm": fsm_result_for_agent2, "feedback": agent4_feedback}, ensure_ascii=False)
                completion_text = agent2_result.get("solidity_code", "")
                contract_stats["tokens"]["agent2"]["prompt"] += calculate_tokens(prompt_text, model_name)
                contract_stats["tokens"]["agent2"]["completion"] += calculate_tokens(completion_text, model_name)
            contract_stats["tokens"]["agent2"]["total"] = (
                contract_stats["tokens"]["agent2"]["prompt"] + contract_stats["tokens"]["agent2"]["completion"]
            )

            if not isinstance(agent2_result, dict) or "solidity_code" not in agent2_result:
                logger.error(
                    f"[{model_name}] Agent2 returned invalid result at iteration {it}: "
                    "Missing 'solidity_code' in result dict."
                )
                break
            final_code = agent2_result["solidity_code"]

            # ---------- Agent3 ----------
            try:
                code_fsm = agent3.run(sanitized_contract_id, final_code)
                transition_count = len(code_fsm.get('fsm', {}).get('transitions', []))
                logger.info(f"[{model_name}] Agent3 成功提取 FSM，包含 {transition_count} 个转换")

                # 统计Agent3的Token
                if hasattr(agent3, "last_token_usage") and isinstance(agent3.last_token_usage, dict):
                    contract_stats["tokens"]["agent3"]["prompt"] += agent3.last_token_usage.get("prompt_tokens", 0)
                    contract_stats["tokens"]["agent3"]["completion"] += agent3.last_token_usage.get("completion_tokens", 0)
                else:
                    prompt_text = final_code
                    completion_text = json.dumps(code_fsm, ensure_ascii=False)
                    contract_stats["tokens"]["agent3"]["prompt"] += calculate_tokens(prompt_text, model_name)
                    contract_stats["tokens"]["agent3"]["completion"] += calculate_tokens(completion_text, model_name)
                contract_stats["tokens"]["agent3"]["total"] = (
                    contract_stats["tokens"]["agent3"]["prompt"] + contract_stats["tokens"]["agent3"]["completion"]
                )
            except Exception as e:
                logger.error(f"[{model_name}] Agent3 提取 FSM 失败：{str(e)}")
                code_fsm = None

            if not isinstance(code_fsm, dict):
                logger.error(
                    f"[{model_name}] Agent3 failed to extract FSM at iteration {it}. "
                    f"Expected dict, got {type(code_fsm)} (value: {code_fsm})"
                )
                break

            # ---------- Agent4 ----------
            try:
                feedback = agent4.run(
                    contract_id=sanitized_contract_id,
                    legal_fsm=legal_fsm,
                    contract_fsm=code_fsm
                )
                logger.info(f"[{model_name}] Agent4 成功生成反馈")

                # 统计Agent4的Token
                if hasattr(agent4, "last_token_usage") and isinstance(agent4.last_token_usage, dict):
                    contract_stats["tokens"]["agent4"]["prompt"] += agent4.last_token_usage.get("prompt_tokens", 0)
                    contract_stats["tokens"]["agent4"]["completion"] += agent4.last_token_usage.get("completion_tokens", 0)
                else:
                    prompt_text = json.dumps({"legal_fsm": legal_fsm, "contract_fsm": code_fsm}, ensure_ascii=False)
                    completion_text = json.dumps(feedback, ensure_ascii=False)
                    contract_stats["tokens"]["agent4"]["prompt"] += calculate_tokens(prompt_text, model_name)
                    contract_stats["tokens"]["agent4"]["completion"] += calculate_tokens(completion_text, model_name)
                contract_stats["tokens"]["agent4"]["total"] = (
                    contract_stats["tokens"]["agent4"]["prompt"] + contract_stats["tokens"]["agent4"]["completion"]
                )
            except Exception as e:
                logger.error(f"[{model_name}] Agent4 分析 FSM 差异失败：{str(e)}")
                feedback = {}

            if not isinstance(feedback, dict):
                logger.warning(f"[{model_name}] Agent4 returned invalid feedback, force continue")
                agent4_feedback = {}
                lifecycle_stats = {
                    "current_issue_count": 0,
                    "new_issue_count": 0,
                    "actionable_issue_count": 0,
                    "repeated_actionable_count": 0,
                    "resolved_issue_count": 0,
                }
                risk_score = 0.0
                risk_issue_count = 0
            else:
                risk_score, risk_issue_count = _compute_feedback_risk_score(
                    feedback,
                    allowed_issue_types=locked_issue_types
                )
                agent4_feedback, lifecycle_stats = _build_actionable_feedback(
                    feedback,
                    issue_lifecycle,
                    iteration_idx=it,
                    max_actionable_issues=(feedback_issue_cap or 12),
                    allowed_issue_types=locked_issue_types,
                )

            current_issue_count = lifecycle_stats["current_issue_count"]
            new_issue_count = lifecycle_stats["new_issue_count"]
            actionable_issue_count = lifecycle_stats["actionable_issue_count"]
            repeated_actionable_count = lifecycle_stats["repeated_actionable_count"]
            resolved_issue_count = lifecycle_stats["resolved_issue_count"]

            if locked_issue_types is None:
                initial_types = _collect_issue_types(agent4_feedback)
                if initial_types:
                    locked_issue_types = set(initial_types)
                    logger.info(
                        f"[{model_name}] Lock issue scope from first iteration: "
                        f"{', '.join(sorted(locked_issue_types))}"
                    )

            if feedback_issue_cap is None and current_issue_count > 0:
                # 锁定修复范围：后续每轮最多处理首轮问题规模，抑制问题集合漂移
                feedback_issue_cap = max(1, min(12, current_issue_count))

            if isinstance(final_code, str) and final_code.strip() and risk_score < best_risk_score:
                best_risk_score = risk_score
                best_code = final_code

            logger.info(
                f"[{model_name}] Agent4 反馈统计：当前问题={current_issue_count}，"
                f"新问题={new_issue_count}，可执行问题={actionable_issue_count}，"
                f"高优先级重复可执行={repeated_actionable_count}，已解决={resolved_issue_count}，"
                f"风险分={risk_score:.2f}（问题数={risk_issue_count}）"
            )

            # ---------- 收敛判断 ----------
            if risk_score <= 0:
                logger.info(f"[{model_name}] FSM aligned at iteration {it} for {contract_id} (no issues found)")
                break

            if previous_risk_score is not None:
                if risk_score > previous_risk_score:
                    logger.warning(
                        f"[{model_name}] Risk score regressed at iteration {it}: "
                        f"{previous_risk_score:.2f} -> {risk_score:.2f}. "
                        "Stop and rollback to best-known code."
                    )
                    if isinstance(best_code, str) and best_code.strip():
                        final_code = best_code
                        fsm_result_for_agent2["previous_contract_code"] = best_code
                    break

                if risk_score < previous_risk_score:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    logger.info(
                        f"[{model_name}] Risk score unchanged at iteration {it} "
                        f"({risk_score:.2f}), stagnant_rounds={no_improve_rounds}"
                    )
                    if no_improve_rounds >= 2:
                        logger.info(
                            f"[{model_name}] Risk score did not improve for {no_improve_rounds} consecutive rounds. "
                            "Stop to avoid ineffective rewrites."
                        )
                        break

            previous_risk_score = risk_score

            if actionable_issue_count == 0:
                logger.info(
                    f"[{model_name}] FSM not aligned at iteration {it}, but no actionable issues under lifecycle policy. "
                    "Stop further modification to avoid over-correction."
                )
                break

            logger.info(f"[{model_name}] FSM not aligned at iteration {it}, actionable issues: {agent4_feedback.get('issues', [])}")
            if it >= MAX_ITER:
                logger.info(f"[{model_name}] Reached max iterations ({MAX_ITER}), stop feedback loop")
                break

    except Exception as e:
        logger.exception(f"[{model_name}] Pipeline crashed for contract {contract_id}: {e}")
    finally:
        _finalize_contract_stats(contract_stats)
        logger.info(
            f"[{model_name}] 合同 {contract_id} 统计：\n"
            f"  耗时：{timedelta(seconds=contract_stats['elapsed_time_seconds'])} \n"
            f"  Token消耗：\n"
            f"    Agent1: 提示词 {contract_stats['tokens']['agent1']['prompt']} | 生成 {contract_stats['tokens']['agent1']['completion']} | 总计 {contract_stats['tokens']['agent1']['total']}\n"
            f"    Agent2: 提示词 {contract_stats['tokens']['agent2']['prompt']} | 生成 {contract_stats['tokens']['agent2']['completion']} | 总计 {contract_stats['tokens']['agent2']['total']}\n"
            f"    Agent3: 提示词 {contract_stats['tokens']['agent3']['prompt']} | 生成 {contract_stats['tokens']['agent3']['completion']} | 总计 {contract_stats['tokens']['agent3']['total']}\n"
            f"    Agent4: 提示词 {contract_stats['tokens']['agent4']['prompt']} | 生成 {contract_stats['tokens']['agent4']['completion']} | 总计 {contract_stats['tokens']['agent4']['total']}\n"
            f"    总计：提示词 {contract_stats['tokens']['total']['prompt']} | 生成 {contract_stats['tokens']['total']['completion']} | 总计 {contract_stats['tokens']['total']['total']}"
        )

    return contract_id, contract_stats

# ===================== 单模型 Pipeline 执行函数 =====================
def run_single_model_pipeline(model_name, contracts):

    # 初始化当前模型的统计数据
    model_stats = {
        "start_time": time.time(),
        "end_time": None,
        "elapsed_time_seconds": 0,
        "tokens": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0
        },
        "contracts": {}  # 按合同ID存储：{contract_id: {time: ..., tokens: ...}}
    }
    global_stats["models"][model_name] = model_stats

    logger.info(f"========== 开始执行模型：{model_name} ==========")
    agent2_output_dir = f"{AGENT2_BASE_OUTPUT_DIR}_{model_name.replace('-', '_').replace('.', '_')}"
    agent1_cache_dir = f"{AGENT1_CACHE_BASE_DIR}_{model_name.replace('-', '_').replace('.', '_')}"
    os.makedirs(agent1_cache_dir, exist_ok=True)


    should_parallel = (
        ENABLE_CONTRACT_PARALLEL
        and len(contracts) > PARALLEL_CONTRACT_THRESHOLD
        and MAX_CONTRACT_WORKERS > 1
    )
    worker_count = max(1, min(MAX_CONTRACT_WORKERS, len(contracts)))

    def _merge_contract_result(cid, cstats):
        model_stats["contracts"][cid] = cstats
        model_stats["tokens"]["total_prompt_tokens"] += cstats["tokens"]["total"]["prompt"]
        model_stats["tokens"]["total_completion_tokens"] += cstats["tokens"]["total"]["completion"]
        model_stats["tokens"]["total_tokens"] += cstats["tokens"]["total"]["total"]

    if should_parallel:
        logger.info(
            f"[{model_name}] 启用合同级并行处理：contracts={len(contracts)}, workers={worker_count}, "
            f"threshold={PARALLEL_CONTRACT_THRESHOLD}"
        )
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{model_name}-contract") as executor:
            future_map = {
                executor.submit(_run_single_contract, model_name, contract, agent2_output_dir, agent1_cache_dir): contract
                for contract in contracts
            }

            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"[{model_name}] Processing Contracts (parallel)"
            ):
                contract = future_map[future]
                fallback_contract_id = contract.get("contract_id", "unknown_contract")
                try:
                    cid, cstats = future.result()
                except Exception as e:
                    logger.exception(f"[{model_name}] Future failed for {fallback_contract_id}: {e}")
                    cid = fallback_contract_id
                    cstats = _init_contract_stats()
                    _finalize_contract_stats(cstats)
                _merge_contract_result(cid, cstats)
    else:
        logger.info(
            f"[{model_name}] 使用串行处理：contracts={len(contracts)}, "
            f"parallel_enabled={ENABLE_CONTRACT_PARALLEL}, workers={MAX_CONTRACT_WORKERS}"
        )
        for contract in tqdm(contracts, desc=f"[{model_name}] Processing Contracts"):
            cid, cstats = _run_single_contract(model_name, contract, agent2_output_dir, agent1_cache_dir)
            _merge_contract_result(cid, cstats)

    # 计算模型总耗时
    model_stats["end_time"] = time.time()
    model_stats["elapsed_time_seconds"] = model_stats["end_time"] - model_stats["start_time"]

    # 记录模型级统计日志
    logger.info(
        f"========== 模型 {model_name} 执行完成 ==========\n"
        f"  总耗时：{timedelta(seconds=model_stats['elapsed_time_seconds'])}\n"
        f"  总Token消耗：\n"
        f"    提示词：{model_stats['tokens']['total_prompt_tokens']}\n"
        f"    生成：{model_stats['tokens']['total_completion_tokens']}\n"
        f"    总计：{model_stats['tokens']['total_tokens']}\n"
        f"  处理合同数：{len(model_stats['contracts'])}\n"
    )

# ===================== 主入口 =====================
def main():
    # 记录全局开始时间
    global_stats["total_start_time"] = time.time()
    logger.info("========== 开始执行多模型Pipeline ==========")

    # 加载数据
    logger.info("Loading dataset...")
    contracts = load_contracts(
        DATASET_PATH,
        limit_one=RUN_ONE_SAMPLE,
        start_index=SAMPLE_START_INDEX,
    )
    logger.info(f"Loaded {len(contracts)} contract(s)")

    if not contracts:
        logger.error("Dataset is empty. Abort.")
        return

    # 多模型遍历执行
    for model_name in MODEL_NAMES:
        run_single_model_pipeline(model_name, contracts)

    # 汇总全局统计
    global_stats["total_end_time"] = time.time()
    global_stats["summary"]["total_time_seconds"] = global_stats["total_end_time"] - global_stats["total_start_time"]
    global_stats["summary"]["processed_contracts"] = len(contracts) * len(MODEL_NAMES)

    for model_name, model_stats in global_stats["models"].items():
        global_stats["summary"]["total_prompt_tokens"] += model_stats["tokens"]["total_prompt_tokens"]
        global_stats["summary"]["total_completion_tokens"] += model_stats["tokens"]["total_completion_tokens"]
        global_stats["summary"]["total_tokens"] += model_stats["tokens"]["total_tokens"]

    # 输出全局统计日志
    logger.info(
        "========== 所有模型Pipeline执行完毕 ==========\n"
        f"  全局总耗时：{timedelta(seconds=global_stats['summary']['total_time_seconds'])}\n"
        f"  全局总Token消耗：\n"
        f"    提示词：{global_stats['summary']['total_prompt_tokens']}\n"
        f"    生成：{global_stats['summary']['total_completion_tokens']}\n"
        f"    总计：{global_stats['summary']['total_tokens']}\n"
        f"  总处理合同数：{global_stats['summary']['processed_contracts']}\n"
        f"  平均每合同耗时：{timedelta(seconds=global_stats['summary']['total_time_seconds'] / global_stats['summary']['processed_contracts']) if global_stats['summary']['processed_contracts'] > 0 else 'N/A'}\n"
        f"  平均每合同Token消耗：{global_stats['summary']['total_tokens'] / global_stats['summary']['processed_contracts'] if global_stats['summary']['processed_contracts'] > 0 else 0:.2f}\n"
    )

    # 保存统计数据到JSON
    save_stats_to_json()

if __name__ == "__main__":
    main()