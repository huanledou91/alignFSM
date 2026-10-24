import hashlib
import json
import logging
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import json5
from llm_router import call_model


# =========================
# 日志配置
# =========================
logger = logging.getLogger("Agent1")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - Agent1 - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

AGENT1_CHUNK_WORKERS = max(1, int(os.getenv("AGENT1_CHUNK_WORKERS", "3")))


def _normalize_constraint_text(text: str) -> str:
    """Normalize constraint text for robust dedup comparison."""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9_./()%\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _constraint_token_set(text: str) -> set:
    """Build comparable token set for near-duplicate detection."""
    stop = {
        "if", "then", "and", "or", "the", "a", "an", "to", "of", "in",
        "for", "with", "under", "where", "that", "this", "is", "are", "be",
        "by", "on", "at", "as", "from", "into", "within", "after", "before",
        "during", "must", "may", "can", "shall", "should"
    }
    norm = _normalize_constraint_text(text)
    tokens = re.findall(r"[a-z0-9_./()%\-]+", norm)
    return {tok for tok in tokens if len(tok) > 2 and tok not in stop}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_constraints(constraints: List[str], similarity_threshold: float = 0.78) -> List[str]:
    """Dedupe constraints by exact normalized form + high token overlap."""
    unique: List[str] = []
    seen_norm = set()
    seen_tokens: List[set] = []

    for c in constraints:
        norm = _normalize_constraint_text(c)
        if not norm:
            continue
        if norm in seen_norm:
            continue

        tok = _constraint_token_set(c)
        is_near_dup = any(_jaccard(tok, prev) >= similarity_threshold for prev in seen_tokens)
        if is_near_dup:
            continue

        unique.append(c)
        seen_norm.add(norm)
        seen_tokens.append(tok)

    return unique

# =========================
# FSM Prompt模板化
# =========================
# 核心任务指令
TASK_TEMPLATE = """
You are a legal-to-code agent. Extract FSM from legal clause AND map to formal logical constraints.
For each transition, explicitly classify:
    - "deontic_type": obligation (MUST), permission (MAY), prohibition (MUST NOT).
    - "temporal_type": timing constraints (e.g., immediate, within_30_days, before_event).
    - "constraint_type": access_control, state_transition, timing_constraint, role_obligation.
Each transition must include:
    - event: short event name in snake_case
    - guard: A pseudo-code formal boolean condition (e.g. "agreement_terms_satisfied == true")
    - action: snake_case_action denoting the legal effect
    - source_span: exact short phrase copied from clause as evidence
    - actors: list of named parties (e.g. ["Agent"])
    - solidity_constraint: draft solidity require statement (e.g. "require(termsAccepted == true)")
Prioritize lifecycle completeness: drafted/active/terminated/violated/expired when applicable.
Output JSON ONLY (no extra text, no explanations).
"""

FORMAT_TEMPLATE = """
{
    "states": [{"label": "SNAKE_CASE_STATE", "description": "brief legal meaning"}],
    "transitions": [
        {
            "from": "ACTIVE",
            "to": "ACTIVE",
            "event": "appointment_accepted",
            "deontic_type": "permission",
            "temporal_type": "immediate",
            "guard": "agreement_terms_satisfied == true",
            "action": "accept_appointment",
            "source_span": "exact phrase from clause",
            "confidence": 0.9,
            "constraint_type": "state_transition",
            "actors": ["Agent"],
            "solidity_constraint": "require(termsAccepted)"
        }
    ],
    "summary_constraints": ["constraint1", "constraint2"]
}
"""

ALLOWED_DEONTIC_TYPES = {
    "obligation",
    "permission",
    "prohibition",
    "state_transition",
}


def _infer_deontic_type(guard: str, action: str) -> str:
    g = (guard or "").lower()
    a = (action or "").lower()
    combined = f"{g} {a}"

    if any(k in combined for k in ["prohibit", "must not", "shall not", "may not", "cannot"]):
        return "prohibition"
    if any(k in combined for k in ["may", "consent", "authorize", "permission", "allow", "right to"]):
        return "permission"
    if any(k in combined for k in ["must", "shall", "oblig", "required", "duty", "agree to"]):
        return "obligation"
    return "state_transition"


def _normalize_actors(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]

    out = []
    seen = set()
    for item in raw:
        actor = str(item or "").strip()
        if not actor:
            continue
        key = actor.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(actor)
    return out


def _normalize_parsed_fsm(parsed: Dict[str, Any]) -> Dict[str, Any]:
    states_in = parsed.get("states", []) if isinstance(parsed, dict) else []
    transitions_in = parsed.get("transitions", []) if isinstance(parsed, dict) else []
    constraints_in = parsed.get("summary_constraints", []) if isinstance(parsed, dict) else []

    states = []
    state_seen = set()
    for s in states_in if isinstance(states_in, list) else []:
        if not isinstance(s, dict):
            continue
        label_raw = str(s.get("label", "")).strip()
        if not label_raw:
            continue
        label = normalize_label(label_raw)
        if label in state_seen:
            continue
        state_seen.add(label)
        states.append({
            "label": label,
            "description": str(s.get("description", "")).strip(),
        })

    transitions = []
    trans_seen = set()
    for t in transitions_in if isinstance(transitions_in, list) else []:
        if not isinstance(t, dict):
            continue

        from_raw = str(t.get("from", "")).strip()
        to_raw = str(t.get("to", "")).strip()
        event = str(t.get("event", "")).strip()
        guard = str(t.get("guard", "")).strip()
        action = str(t.get("action", "")).strip()
        source_span = str(t.get("source_span", t.get("legal_reference", ""))).strip()
        confidence = t.get("confidence", 0.0)
        solidity_constraint = str(t.get("solidity_constraint", "")).strip()
        deontic_type = str(t.get("deontic_type", "")).strip().lower()
        temporal_type = str(t.get("temporal_type", "")).strip()
        constraint_type = str(t.get("constraint_type", "")).strip()

        if not from_raw and not to_raw:
            continue
        if not from_raw:
            from_raw = to_raw
        if not to_raw:
            to_raw = from_raw

        if not (guard or action or event):
            continue

        from_label = normalize_label(from_raw)
        to_label = normalize_label(to_raw)
        if not event:
            event = "auto_transition"

        if deontic_type not in ALLOWED_DEONTIC_TYPES:
            deontic_type = _infer_deontic_type(guard, action)

        sig = _normalize_constraint_text(f"{from_label}|{to_label}|{event}|{guard}|{action}")
        if sig in trans_seen:
            continue
        trans_seen.add(sig)

        transitions.append({
            "from": from_label,
            "to": to_label,
            "event": event,
            "guard": guard,
            "action": action,
            "source_span": source_span,
            "confidence": confidence,
            "solidity_constraint": solidity_constraint,
            "deontic_type": deontic_type,
            "temporal_type": temporal_type,
            "constraint_type": constraint_type,
            "actors": _normalize_actors(t.get("actors", [])),
        })

        if from_label not in state_seen:
            states.append({"label": from_label, "description": ""})
            state_seen.add(from_label)
        if to_label not in state_seen:
            states.append({"label": to_label, "description": ""})
            state_seen.add(to_label)

    constraints = []
    if isinstance(constraints_in, list):
        constraints = [str(c).strip() for c in constraints_in if str(c).strip()]

    return {
        "states": states,
        "transitions": transitions,
        "summary_constraints": constraints,
    }


def _fsm_score(parsed: Dict[str, Any]) -> int:
    states = parsed.get("states", []) if isinstance(parsed, dict) else []
    transitions = parsed.get("transitions", []) if isinstance(parsed, dict) else []
    constraints = parsed.get("summary_constraints", []) if isinstance(parsed, dict) else []
    return len(transitions) * 10 + len(states) * 2 + len(constraints)


def build_retry_prompt(clause_text: str, retry_idx: int) -> str:
    repair_instructions = f"""
=== STRICT REPAIR MODE (attempt {retry_idx + 1}) ===
- Previous output may be invalid or empty. Produce valid JSON only.
- Do not output markdown or explanation.
- Ensure transitions are not empty when the clause contains obligations, dates, termination, renewal, payment, breach, or restrictions.
- Include event and source_span for every transition.
- Use concise but semantically faithful states and transitions.
"""
    return build_prompt(clause_text) + "\n" + repair_instructions.strip()

def _coerce_to_fsm_dict(parsed: Any) -> Optional[Dict[str, Any]]:
    """Coerce parsed JSON root into FSM dict shape when possible."""
    if isinstance(parsed, dict):
        # Common wrappers seen in upstream pipelines.
        for key in ("legal_fsm", "fsm", "data"):
            if isinstance(parsed.get(key), dict):
                inner = parsed[key]
                if any(k in inner for k in ("states", "transitions", "summary_constraints")):
                    return inner
        return parsed

    if isinstance(parsed, list):
        items = [x for x in parsed if isinstance(x, dict)]
        if not items:
            return None

        transition_like = sum(
            1 for x in items if any(k in x for k in ("from", "to", "event", "guard", "action"))
        )
        state_like = sum(1 for x in items if "label" in x and "from" not in x and "to" not in x)

        if transition_like >= state_like:
            return {
                "states": [],
                "transitions": items,
                "summary_constraints": [],
            }

        return {
            "states": items,
            "transitions": [],
            "summary_constraints": [],
        }

    return None


def _load_json_relaxed(text: str) -> Tuple[Optional[Any], str]:
    """Best-effort JSON loader: json -> json5 -> literal_eval style repair."""
    try:
        return json.loads(text), text
    except Exception:
        pass

    try:
        return json5.loads(text), text
    except Exception:
        pass

    repaired = re.sub(r"(?<!\\)'", '"', text)
    repaired2 = repaired.replace("true", "True").replace("false", "False").replace("null", "None")
    try:
        import ast
        return ast.literal_eval(repaired2), repaired2
    except Exception:
        return None, text


def safe_parse_json(text: str) -> Tuple[Optional[dict], str]:
    """尝试解析给定字符串为dict。

    1. 直接用标准库 `json.loads`。
    2. 若失败，则用 `json5.loads`（支持单引号、尾随逗号、不加引号的键等）。
    3. 再失败时尝试简单正则修复（双引号替换、true/false/null 替换）。

    返回 (parsed_dict, used_text)。若无法解析，则 parsed_dict 为 None，
    used_text 为最后尝试的字符串（便于日志记录）。
    """
    parsed, used_text = _load_json_relaxed(text)
    coerced = _coerce_to_fsm_dict(parsed)
    if coerced is not None:
        return coerced, used_text

    # 从混杂文本中再尝试提取最大对象/数组片段
    candidates = []
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1].strip())

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end + 1].strip())

    # 优先尝试更长片段，避免截到内层局部结构
    for candidate in sorted(candidates, key=len, reverse=True):
        parsed_candidate, used_candidate = _load_json_relaxed(candidate)
        coerced_candidate = _coerce_to_fsm_dict(parsed_candidate)
        if coerced_candidate is not None:
            return coerced_candidate, used_candidate

    return None, text

def validate_json_structure(json_str: str) -> bool:
    if not json_str or not json_str.strip():
        return False
    
    # 检查括号匹配
    stack = []
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if not stack:
                return False
            last = stack.pop()
            if (char == '}' and last != '{') or (char == ']' and last != '['):
                return False
    
    return len(stack) == 0
def clean_model_response(response_text: str) -> str:
    """
    清洗模型返回的文本，去除无效格式，保留核心JSON内容。
    不要在这里将有效输出扔掉，若无法保证JSON结构则仍返回原始文本，
    由调用方负责进一步解析/修复。
    """
    if not isinstance(response_text, str):
        return ""

    # 步骤1：去除首尾空白字符
    cleaned = response_text.strip()
    if not cleaned:
        return ""

    # 步骤2：去除Markdown JSON代码块标记（最常见问题）
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    if not cleaned:
        return ""

    # 步骤3：优先保留可解析的完整对象/数组，不再强制裁剪为 {...}
    parsed_direct, _ = _load_json_relaxed(cleaned)
    if parsed_direct is not None:
        return cleaned

    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")

    candidates = []
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(cleaned[obj_start:obj_end + 1].strip())
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(cleaned[arr_start:arr_end + 1].strip())

    for candidate in sorted(candidates, key=len, reverse=True):
        parsed_candidate, _ = _load_json_relaxed(candidate)
        if parsed_candidate is not None:
            return candidate

    # 步骤4：轻量修复（尾随逗号）
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    parsed_repaired, _ = _load_json_relaxed(cleaned)
    if parsed_repaired is not None:
        return cleaned

    logger.warning(f"Cleaned response doesn't have proper JSON structure: {cleaned[:100]}...")

    return cleaned


# 动态构建Prompt（核心：只拼接必要内容，长度减少60%）
def build_prompt(clause_text: str) -> str:
    prompt_parts = [
        TASK_TEMPLATE.strip(),
        "\n=== OUTPUT FORMAT ===",
        FORMAT_TEMPLATE.strip(),
        "\n=== CLAUSE TO PROCESS ===",
        f"Clause:\n{clause_text.strip()}"
    ]
    final_prompt = "\n".join([p for p in prompt_parts if p.strip()])
    return final_prompt


def _heuristic_fsm_from_clause(clause_text: str) -> Dict[str, Any]:
    """Non-LLM fallback to avoid returning completely empty FSM."""
    text = str(clause_text or "").strip()
    if not text:
        return {"states": [], "transitions": [], "summary_constraints": []}

    states = [{"label": "ACTIVE", "description": "Agreement is active"}]
    transitions = []

    lower = text.lower()
    if any(k in lower for k in ["sign", "effective", "entered into"]):
        states.append({"label": "DRAFTED", "description": "Agreement drafted"})
        transitions.append({
            "from": "DRAFTED",
            "to": "ACTIVE",
            "event": "sign_agreement",
            "guard": "Parties sign the agreement",
            "action": "Agreement becomes effective",
            "source_span": "sign/effective clause",
            "confidence": 0.4,
            "solidity_constraint": "",
            "constraint_type": "state_transition",
            "actors": [],
        })

    if any(k in lower for k in ["renew", "extension"]):
        transitions.append({
            "from": "ACTIVE",
            "to": "ACTIVE",
            "event": "renew_agreement",
            "guard": "Renewal conditions are satisfied",
            "action": "Agreement term is extended",
            "source_span": "renewal clause",
            "confidence": 0.35,
            "solidity_constraint": "",
            "constraint_type": "timing_constraint",
            "actors": [],
        })

    if any(k in lower for k in ["terminate", "termination", "expire", "expiration"]):
        states.append({"label": "TERMINATED", "description": "Agreement terminated"})
        transitions.append({
            "from": "ACTIVE",
            "to": "TERMINATED",
            "event": "terminate_agreement",
            "guard": "Termination or expiration condition is met",
            "action": "Agreement is terminated",
            "source_span": "termination clause",
            "confidence": 0.4,
            "solidity_constraint": "",
            "constraint_type": "state_transition",
            "actors": [],
        })

    if any(k in lower for k in ["breach", "violate", "prohibit", "forbid"]):
        states.append({"label": "VIOLATED", "description": "Contract violation detected"})
        transitions.append({
            "from": "ACTIVE",
            "to": "VIOLATED",
            "event": "detect_violation",
            "guard": "Breach or prohibited action occurs",
            "action": "Record violation and trigger remedies",
            "source_span": "breach/violation clause",
            "confidence": 0.4,
            "solidity_constraint": "",
            "constraint_type": "access_control",
            "actors": [],
        })

    if any(k in lower for k in ["pay", "payment", "fee", "revenue", "royalty"]):
        transitions.append({
            "from": "ACTIVE",
            "to": "ACTIVE",
            "event": "process_payment",
            "guard": "Payment trigger condition is met",
            "action": "Calculate and settle payment obligations",
            "source_span": "payment/revenue clause",
            "confidence": 0.35,
            "solidity_constraint": "",
            "constraint_type": "role_obligation",
            "actors": [],
        })

    normalized = _normalize_parsed_fsm({
        "states": states,
        "transitions": transitions,
        "summary_constraints": [
            f"if {t.get('guard')}, then {t.get('action')}" for t in transitions if t.get("guard") and t.get("action")
        ],
    })
    return normalized


def _extract_full_contract_text(contract_data: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    text_obj = contract_data.get("text", "")
    if isinstance(text_obj, dict):
        full = str(text_obj.get("full", "")).strip()
        if full:
            return full

    chunk_texts = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        t = str(c.get("text", "")).strip()
        if t:
            chunk_texts.append(t)
    return "\n\n".join(chunk_texts)


def extract_fsm_from_full_text(full_text: str, model_name: str) -> Dict[str, Any]:
    if not str(full_text or "").strip():
        return {"states": [], "transitions": [], "summary_constraints": []}

    prompt = build_retry_prompt(full_text, retry_idx=99)
    try:
        resp_text = call_model(
            model_name=model_name,
            prompt=prompt,
            temperature=0.0,
            expect_json=True
        )
        cleaned_resp = clean_model_response(resp_text)
        parsed, _ = safe_parse_json(cleaned_resp)
        if parsed is not None:
            return _normalize_parsed_fsm(parsed)
    except Exception as e:
        logger.warning(f"Full-text recovery extraction failed: {e}")

    return _heuristic_fsm_from_clause(full_text)


# =========================
# FSM 工具函数
# =========================
def normalize_label(label: str) -> str:
    """优化：确保标签是代码友好的snake_case"""
    label = label.strip().replace(" ", "_").replace("-", "_").upper()
    return label


def stable_state_id(label: str) -> str:
    h = hashlib.md5(label.lower().encode()).hexdigest()[:8]
    return f"S_{h}"


def extract_fsm_from_chunk(chunk: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """单 chunk → FSM"""
    # 过滤空文本
    chunk_text = chunk.get("text", "").strip()
    chunk_id = chunk.get("chunk_id", "unknown")
    if not chunk_text:
        logger.warning(f"Empty chunk text (chunk_id: {chunk_id})")
        return {"states": [], "transitions": [], "summary_constraints": []}

    max_retries = 3
    best_candidate = None
    best_score = -1

    for retry in range(max_retries):
        try:
            prompt = build_prompt(chunk_text) if retry == 0 else build_retry_prompt(chunk_text, retry)
            logger.debug(f"Built prompt length: {len(prompt)} chars (chunk_id: {chunk_id}, retry={retry + 1})")

            # ==========适配 call_model 返回的纯文本格式 ==========
            resp_text = call_model(
                model_name=model_name,
                prompt=prompt,
                temperature=0.0,
                expect_json=True
            )

            if not isinstance(resp_text, str):
                resp_text = str(resp_text or "")

            cleaned_resp = clean_model_response(resp_text)
            logger.info(f"Raw response (chunk_id: {chunk_id}, length: {len(resp_text)}): {repr(resp_text[:300])}")
            logger.info(f"Cleaned response (chunk_id: {chunk_id}): '{cleaned_resp}' (length: {len(cleaned_resp)})")
            if not resp_text.strip():
                logger.error(
                    f"Model returned empty text before cleaning (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id})"
                )
            if not cleaned_resp:
                logger.error(
                    f"Response is empty after cleaning (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id})")
                continue
            # 简单检查：结束括号
            if len(cleaned_resp) < 10 or not cleaned_resp.strip().endswith(('}', ']')):
                logger.warning(
                    f"Response appears incomplete (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id}): "
                    "too short or doesn't end with '}'/']'"
                )


            # ========== JSON 解析+严格兜底 ==========
            parsed, used = safe_parse_json(cleaned_resp)
            if parsed is None:
                logger.error(f"Unable to parse JSON after cleaning (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id}). "
                             f"Used string: {used[:200]}...")
                continue

            normalized = _normalize_parsed_fsm(parsed)
            score = _fsm_score(normalized)
            if score > best_score:
                best_score = score
                best_candidate = normalized

            if len(normalized.get("transitions", [])) >= 1 and len(normalized.get("states", [])) >= 2:
                logger.info(
                    f"Extracted FSM for chunk {chunk_id}: {len(normalized['transitions'])} transitions, {len(normalized['summary_constraints'])} constraints")
                return normalized

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id}): {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error (retry {retry + 1}/{max_retries}, chunk_id: {chunk_id}): {str(e)}")
            continue

    if best_candidate is not None and _fsm_score(best_candidate) > 0:
        logger.warning(
            f"Using best partial FSM candidate for chunk {chunk_id}: "
            f"{len(best_candidate.get('states', []))} states, {len(best_candidate.get('transitions', []))} transitions"
        )
        return best_candidate

    logger.error(f"All retries failed for chunk {chunk_id}, fallback to heuristic extraction")
    return _heuristic_fsm_from_clause(chunk_text)


def merge_fsm_parts(fsm_parts, contract_uid, model_name):
    state_map = {}
    transitions = []
    evidence_map = {}
    seen_transition_signatures = set()

    for part in fsm_parts:
        for s in part.get("states", []):
            label_raw = str(s.get("label", "")).strip() if isinstance(s, dict) else ""
            if not label_raw:
                continue
            label = normalize_label(label_raw)
            sid = stable_state_id(label)
            if sid not in state_map:
                state_map[sid] = {
                    "state_id": sid,
                    "label": label,
                    "description": s.get("description", "")
                }

        for t in part.get("transitions", []):
            from_raw = str(t.get("from", "")).strip()
            to_raw = str(t.get("to", "")).strip()
            event = str(t.get("event", "")).strip()
            guard = str(t.get("guard", "")).strip()
            action = str(t.get("action", "")).strip()
            source_span = str(t.get("source_span", "")).strip()
            actors = _normalize_actors(t.get("actors", []))

            if not from_raw and not to_raw:
                continue
            if not from_raw:
                from_raw = to_raw
            if not to_raw:
                to_raw = from_raw

            from_label = normalize_label(from_raw)
            to_label = normalize_label(to_raw)

            from_id = stable_state_id(from_label)
            to_id = stable_state_id(to_label)

            if from_id not in state_map:
                state_map[from_id] = {
                    "state_id": from_id,
                    "label": from_label,
                    "description": "",
                }
            if to_id not in state_map:
                state_map[to_id] = {
                    "state_id": to_id,
                    "label": to_label,
                    "description": "",
                }

            signature = _normalize_constraint_text(f"{from_id}|{to_id}|{event}|{guard}|{action}")
            if signature in seen_transition_signatures:
                continue
            seen_transition_signatures.add(signature)

            tid = hashlib.md5(f"{from_id}->{to_id}:{event}:{guard}:{action}".encode()).hexdigest()[:10]

            transitions.append({
                "transition_id": tid,
                "from": from_id,
                "to": to_id,
                "event": event,
                "guard": guard,
                "action": action,
                "constraint_type": t.get("constraint_type", "state_transition"),
                "actors": actors,
            })

            evidence_map[tid] = {
                "confidence": t.get("confidence"),
                "solidity_constraint": t.get("solidity_constraint"),
                "chunk_id": t.get("chunk_id"),
                "source_span": source_span,
            }

    return {
        "fsm_id": contract_uid,
        "model": model_name,
        "legal_fsm": {
            "states": list(state_map.values()),
            "transitions": transitions
        },
        "evidence": evidence_map
    }


def _merged_fsm_score(merged: Dict[str, Any]) -> int:
    legal = merged.get("legal_fsm", {}) if isinstance(merged, dict) else {}
    states = legal.get("states", []) if isinstance(legal, dict) else []
    transitions = legal.get("transitions", []) if isinstance(legal, dict) else []
    return len(transitions) * 10 + len(states) * 2

class Agent1FSM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def run(self, contract_data: Dict[str, Any]) -> Dict[str, Any]:
        contract_uid = contract_data.get("uid", contract_data.get("contract_id", "unknown"))
        text_field = contract_data.get("text", {})
        chunks = text_field.get("chunks", []) if isinstance(text_field, dict) else []


        if not isinstance(chunks, list):

            text_str = str(contract_data.get("text", ""))

            chunks = []
            paragraphs = [p.strip() for p in text_str.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):

                if len(para) > 300:
                    sub_paras = [para[j:j + 300] for j in range(0, len(para), 300)]
                    for k, sub_para in enumerate(sub_paras):
                        chunks.append({"chunk_id": f"chunk_{i}_{k}", "text": sub_para})
                else:
                    chunks.append({"chunk_id": f"chunk_{i}", "text": para})

        logger.info(f"Processing contract {contract_uid}: {len(chunks)} chunks total")
        full_text = _extract_full_contract_text(contract_data, chunks)

        fsm_parts: List[Dict[str, Any]] = []
        worker_count = min(AGENT1_CHUNK_WORKERS, max(1, len(chunks)))

        if worker_count > 1 and len(chunks) > 1:
            logger.info(
                f"Agent1 chunk extraction in parallel: workers={worker_count}, chunks={len(chunks)}"
            )
            indexed_parts: Dict[int, Dict[str, Any]] = {}

            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="agent1-chunk") as executor:
                future_map = {
                    executor.submit(extract_fsm_from_chunk, chunk, self.model_name): (idx, chunk)
                    for idx, chunk in enumerate(chunks)
                }

                for future in as_completed(future_map):
                    idx, chunk = future_map[future]
                    try:
                        frag = future.result()
                    except Exception as e:
                        chunk_text = str(chunk.get("text", ""))
                        logger.error(f"Chunk extraction failed (chunk_id={chunk.get('chunk_id')}): {e}")
                        frag = _heuristic_fsm_from_clause(chunk_text)

                    for t in frag.get("transitions", []):
                        t["chunk_id"] = chunk.get("chunk_id")
                    indexed_parts[idx] = frag

            fsm_parts = [indexed_parts[i] for i in range(len(chunks)) if i in indexed_parts]
        else:
            for chunk in chunks:
                frag = extract_fsm_from_chunk(chunk, self.model_name)

                for t in frag.get("transitions", []):
                    t["chunk_id"] = chunk.get("chunk_id")
                fsm_parts.append(frag)

        merged = merge_fsm_parts(fsm_parts, contract_uid, self.model_name)

        # Sparse-output recovery: retry one full-text extraction when chunk-level result is too weak.
        if _merged_fsm_score(merged) < 14 and full_text:
            recovered_part = extract_fsm_from_full_text(full_text, self.model_name)
            for t in recovered_part.get("transitions", []):
                t.setdefault("chunk_id", "full_text_recovery")

            recovered_merged = merge_fsm_parts([recovered_part], contract_uid, self.model_name)
            if _merged_fsm_score(recovered_merged) > _merged_fsm_score(merged):
                logger.info(
                    f"Using full-text recovered FSM for {contract_uid}: "
                    f"score {_merged_fsm_score(merged)} -> {_merged_fsm_score(recovered_merged)}"
                )
                merged = recovered_merged

        # ensure basic keys exist
        merged["states"] = merged.get("states", [])
        merged["transitions"] = merged.get("transitions", [])

        # --- 从 legal_fsm.transitions 中提取业务约束，并做去重归并 ---
        constraint_dicts = []
        for t in merged.get("legal_fsm", {}).get("transitions", []):
            guard = t.get("guard", "").strip()
            action = t.get("action", "").strip()
            ctype = t.get("constraint_type", "state_transition")
            actors = t.get("actors") or []
            if guard and action:
                text = f"if {guard}, then {action}"
            elif guard:
                text = guard
            elif action:
                text = action
            else:
                continue
            enforcement = "off_chain" if ctype == "off_chain" else "on_chain"
            constraint_dicts.append({
                "constraint": text,
                "type": ctype,
                "actors": actors,
                "enforcement": enforcement
            })

        raw_count = len(constraint_dicts)
        # Dedup by constraint text, preserving first occurrence with full metadata
        deduped_texts_set = set(_dedupe_constraints([c["constraint"] for c in constraint_dicts]))
        seen_texts_set: set = set()
        constraints = []
        for c in constraint_dicts:
            if c["constraint"] in deduped_texts_set and c["constraint"] not in seen_texts_set:
                seen_texts_set.add(c["constraint"])
                constraints.append(c)
        merged["all_solidity_constraints"] = constraints
        merged["constraint_count"] = len(constraints)
        logger.info(f"Constraint dedup merged {raw_count} -> {len(constraints)}")

        logger.info(f"[${self.model_name}] Agent1 返回的原始数据：{json.dumps(merged, ensure_ascii=False, indent=2)}")
        logger.info(f"[${self.model_name}] Agent1 返回数据的键：{list(merged.keys())}")

        return merged


