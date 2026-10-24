import os
import json
import re
import hashlib
import logging
from typing import Dict, Any, List, Optional
from llm_router import call_model
from solidity_compiler import get_compiler

# =========================
# 日志配置
# =========================
logger = logging.getLogger("Agent2")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - Agent2 - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Solidity 核心配置常量
SOLIDITY_VERSION = "^0.8.20"

# 默认高风险模式（用于输出清洗）
HIGH_RISK_PATTERNS = [
    r"\btx\.origin\b",
    r"\bselfdestruct\s*\(",
    r"\bdelegatecall\s*\(",
]

# 默认安全规则（当前业务主流程可关闭，保留可扩展入口）
SECURITY_RULES: List[str] = []

# =========================

# =========================
def sanitize_filename(filename: str) -> str:
    """
    清理文件名中的非法字符（适配 Windows/macOS/Linux）
    :param filename: 原始文件名
    :return: 合法的文件名
    """
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized

# =========================
# 工具函数：代码提取与清洗
# =========================
def _extract_and_clean_solidity_code(llm_output: Any) -> str:
    """
    提取并深度清洗Solidity代码，解决格式错乱问题
    :param llm_output: LLM原始输出
    :return: 纯净、格式规整的Solidity代码
    """
    # 步骤1：提取原始文本
    if isinstance(llm_output, dict):
        llm_text = llm_output.get("text", "")
    else:
        llm_text = str(llm_output)

    # 步骤2：提取Solidity代码块（兼容多种格式）
    code = ""
    fenced_solidity = re.search(r"```solidity\s*(.*?)```", llm_text, re.DOTALL | re.IGNORECASE)
    fenced_generic = re.search(r"```\s*(.*?)```", llm_text, re.DOTALL | re.IGNORECASE)
    pragma_index = llm_text.lower().find("pragma solidity")

    if fenced_solidity:
        code = fenced_solidity.group(1).strip()
    elif fenced_generic:
        code = fenced_generic.group(1).strip()
    elif pragma_index != -1:
        code = llm_text[pragma_index:].strip()

    # 若未提取到代码块，直接使用清洗后的原始文本
    if not code:
        code = llm_text.strip()

    # 步骤3：深度清洗（核心：解决格式错乱）
    # 3.1 去除所有非代码相关的文本说明
    code = re.sub(r"//.*以下是.*|//.*以上是.*|===.*===|---.*---", "", code)
    # 3.2 去除多余空行、制表符，规整格式
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = re.sub(r"\t", "    ", code)
    code = re.sub(r" +", " ", code)

    # 3.2.1 移除过度冗长的说明性注释，避免污染后续行为抽取
    filtered_lines = []
    noisy_comment_markers = (
        "semantic alignment",
        "in a real scenario",
        "for this exercise",
        "placeholder",
        "this function is intended",
    )
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("//"):
            ls = s.lower()
            if len(s) > 160:
                continue
            if any(m in ls for m in noisy_comment_markers):
                continue
        filtered_lines.append(ln)
    code = "\n".join(filtered_lines)
    # 3.3 确保pragma指令存在（编译必备）
    if not re.search(r"pragma solidity", code, re.IGNORECASE):
        code = f"pragma solidity {SOLIDITY_VERSION};\n\n" + code
    # 3.4 闭合未完成的大括号
    brace_count = code.count("{") - code.count("}")
    if brace_count > 0:
        code += "\n" + "}" * brace_count
    elif brace_count < 0:
        code = code[:code.rindex("}") * (abs(brace_count) + 1)]
    # 3.5 去除高危语法的注释残留（避免干扰编译）
    for pattern in HIGH_RISK_PATTERNS:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE)

    return code.strip()


def _looks_like_valid_contract_code(code: str) -> bool:
    """Solidity有效性快速判断，用于骨架/生成结果兜底。"""
    if not code or len(code.strip()) < 40:
        return False
    lowered = code.lower()
    if "pragma solidity" not in lowered or "function " not in lowered:
        return False

    # 必须包含显式的合约声明块，避免接受 "pragma ... { ... }" 这类坏输出
    if not re.search(r"\bcontract\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", code):
        return False

    # 基本括号平衡检查
    if code.count("{") == 0 or code.count("{") != code.count("}"):
        return False

    return True


def _extract_structure_metrics(code: str) -> Dict[str, int]:
    """提取可被 Agent3 识别的最小结构指标（仅结构，不含语义硬编码）。"""
    if not code:
        return {
            "enum_count": 0,
            "state_var_count": 0,
            "state_transition_assignments": 0,
            "state_guards": 0,
            "function_count": 0,
        }

    enum_count = len(re.findall(r"\benum\s+\w+\s*\{", code, re.IGNORECASE))
    state_var_count = len(re.findall(r"\b\w+\s+public\s+\w+\s*;", code, re.IGNORECASE))
    state_transition_assignments = len(re.findall(r"\b\w+\s*=\s*\w+\.\w+", code, re.IGNORECASE))
    state_guards = len(re.findall(r"require\s*\(\s*\w+\s*==\s*\w+\.\w+", code, re.IGNORECASE))
    function_count = len(re.findall(r"\bfunction\s+\w+\s*\(", code, re.IGNORECASE))

    return {
        "enum_count": enum_count,
        "state_var_count": state_var_count,
        "state_transition_assignments": state_transition_assignments,
        "state_guards": state_guards,
        "function_count": function_count,
    }


def _meets_min_fsm_shape(code: str) -> bool:
    """仅检查代码是否具备最小状态机结构，避免 Agent3 提取为空。"""
    m = _extract_structure_metrics(code)
    return (
        m["function_count"] >= 2 and
        m["enum_count"] >= 1 and
        m["state_transition_assignments"] >= 1 and
        m["state_guards"] >= 1
    )


# =========================
# 多角色识别与 Solidity 声明生成
# =========================
_SKIP_ROLES = {"owner", "msgsender", "the", "a", "an", "parties", "party", "user"}

def _normalize_role_name(actor: str) -> str:
    """Convert actor/party name to a valid camelCase Solidity identifier."""
    parts = re.split(r"[\s_\-/]+", actor.strip().lower())
    parts = [p for p in parts if p and len(p) > 1]
    if not parts:
        return ""
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

def _extract_roles_from_constraints(constraints: List) -> List[str]:
    """Extract unique camelCase role names from typed constraint dicts (skips 'owner')."""
    seen: set = set()
    roles: List[str] = []
    for c in constraints:
        if not isinstance(c, dict):
            continue
        for actor in (c.get("actors") or []):
            norm = _normalize_role_name(actor)
            if not norm or norm in _SKIP_ROLES or norm in seen:
                continue
            seen.add(norm)
            roles.append(norm)
    return roles

def _build_role_declarations(roles: List[str]) -> str:
    """Generate Solidity state variable declarations for each identified role."""
    lines = []
    for role in roles:
        if not role:
            continue
        is_single = any(kw in role.lower() for kw in ["agent", "counsel", "admin", "issuer", "pss", "notary", "broker"])
        if is_single:
            lines.append(f"    address public {role};")
        else:
            cap = role[0].upper() + role[1:]
            lines.append(f"    mapping(address => bool) public is{cap};")
    return "\n".join(lines)

def _build_fallback_skeleton(contract_name: str, include_security_rules: bool, roles: List[str] = None) -> str:
    """本地兜底骨架：仅保留中性FSM结构，避免注入与法律文本无关的业务动作。"""
    role_decls = _build_role_declarations(roles or [])
    role_section = f"\n{role_decls}" if role_decls else ""
    return f"""pragma solidity {SOLIDITY_VERSION};

contract {contract_name} {{
    address public owner;{role_section}

    enum ContractState {{ STATE_0, STATE_1, STATE_2 }}
    ContractState public currentState;

    event StateTransitioned(ContractState indexed fromState, ContractState indexed toState);

    constructor() {{
        owner = msg.sender;
        currentState = ContractState.STATE_0;
    }}

    function doState1Transition() external {{
        require(currentState == ContractState.STATE_0, "invalid state");
        ContractState prev = currentState;
        currentState = ContractState.STATE_1;
        emit StateTransitioned(prev, ContractState.STATE_1);
    }}

    function doState2Transition() external {{
        require(currentState == ContractState.STATE_1, "invalid state");
        ContractState prev = currentState;
        currentState = ContractState.STATE_2;
        emit StateTransitioned(prev, ContractState.STATE_2);
    }}
}}
""".strip()


def _build_fallback_fsm_scaffold(contract_name: str) -> str:
    """结构兜底：仅生成通用FSM框架，不绑定任何法律状态名。"""
    return f"""pragma solidity {SOLIDITY_VERSION};

contract {contract_name} {{
    address public owner;

    enum State {{ STATE_0, STATE_1, STATE_2 }}
    State public currentState;

    constructor() {{
        owner = msg.sender;
        currentState = State.STATE_0;
    }}

    function transitionToState1() external {{
        require(currentState == State.STATE_0, "invalid state");
        currentState = State.STATE_1;
    }}

    function transitionToState2() external {{
        require(currentState == State.STATE_1, "invalid state");
        currentState = State.STATE_2;
    }}
}}
""".strip()


def _build_skeleton_from_legal_fsm(
    contract_name: str,
    legal_fsm: Dict[str, Any],
    roles: List[str],
    include_security_rules: bool,
) -> str:
    """Deterministically build a Solidity skeleton using Agent1's FSM states & transitions.

    This replaces the model-generated Stage 1 skeleton, guaranteeing correct enum and
    transition stubs even when the model would otherwise ignore FSM structure.
    """
    states_list = legal_fsm.get("states", [])
    transitions_list = legal_fsm.get("transitions", [])

    # Build ordered, deduplicated state enum (max 14 labels)
    state_labels: List[str] = []
    seen_lbl: set = set()
    for s in states_list:
        lbl = (s.get("label") or "").strip()
        if lbl and lbl not in seen_lbl and len(state_labels) < 14:
            state_labels.append(lbl)
            seen_lbl.add(lbl)
    if not state_labels:
        state_labels = ["STATE_INITIAL", "STATE_ACTIVE", "STATE_COMPLETED"]
    initial_state = state_labels[0]

    id_to_label = {
        s["state_id"]: s["label"]
        for s in states_list
        if s.get("state_id") and s.get("label")
    }

    # Build one transition function per unique from→to pair (max 10)
    func_stubs: List[str] = []
    seen_pairs: set = set()
    for t in transitions_list:
        from_label = id_to_label.get(t.get("from", ""), "")
        to_label = id_to_label.get(t.get("to", ""), "")
        if not from_label or not to_label:
            continue
        pair = (from_label, to_label)
        if pair in seen_pairs or len(seen_pairs) >= 10:
            continue
        seen_pairs.add(pair)
        camel_to = "".join(w.capitalize() for w in to_label.split("_"))
        guard_hint = (t.get("guard") or "")[:80]
        func_stubs.append(
            f"    function do{camel_to}() external {{\n"
            f"        require(currentState == ContractState.{from_label}, \"Must be in {from_label} state\");\n"
            f"        // Guard: {guard_hint}\n"
            f"        ContractState prev = currentState;\n"
            f"        currentState = ContractState.{to_label};\n"
            f"        emit StateTransitioned(prev, ContractState.{to_label});\n"
            f"    }}"
        )
    if not func_stubs:
        next_state = state_labels[min(1, len(state_labels) - 1)]
        func_stubs = [
            f"    function doInitialTransition() external {{\n"
            f"        require(currentState == ContractState.{initial_state}, \"Must be in initial state\");\n"
            f"        ContractState prev = currentState;\n"
            f"        currentState = ContractState.{next_state};\n"
            f"        emit StateTransitioned(prev, ContractState.{next_state});\n"
            f"    }}"
        ]

    enum_body = "\n        ".join(state_labels)
    enum_body = ",\n        ".join(state_labels)
    role_decls = _build_role_declarations(roles)
    role_section = f"\n{role_decls}" if role_decls else ""
    funcs_code = "\n\n".join(func_stubs)

    return (
        f"// SPDX-License-Identifier: MIT\n"
        f"pragma solidity {SOLIDITY_VERSION};\n\n"
        f"contract {contract_name} {{\n"
        f"    address public owner;\n"
        f"    mapping(address => bool) public isAffiliate;\n"
        f"{role_section}\n\n"
        f"    enum ContractState {{\n"
        f"        {enum_body}\n"
        f"    }}\n"
        f"    ContractState public currentState;\n\n"
        f"    event StateTransitioned(ContractState indexed fromState, ContractState indexed toState);\n\n"
        f"    constructor() {{\n"
        f"        owner = msg.sender;\n"
        f"        currentState = ContractState.{initial_state};\n"
        f"    }}\n\n"
        f"{funcs_code}\n"
        f"}}"
    )

def _compact_code_for_prompt(code: str, max_chars: int = 4200) -> str:
    """Compress Solidity code for prompt context to reduce token usage.

    Notes:
    - This function is only for prompt context, not for persisted output code.
    - It removes comments and excessive whitespace.
    - If still too long, it keeps head+tail to preserve global structure hints.
    """
    if not code:
        return ""

    compact = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    compact = re.sub(r"//[^\n]*", "", compact)
    compact = re.sub(r"\n\s*\n+", "\n", compact)
    compact = re.sub(r"[ \t]+", " ", compact)
    compact = compact.strip()

    if len(compact) <= max_chars:
        return compact

    head_len = int(max_chars * 0.65)
    tail_len = int(max_chars * 0.30)
    head = compact[:head_len]
    tail = compact[-tail_len:]
    return head + "\n// ... context truncated for token budget ...\n" + tail


def _extract_function_signatures(code: str) -> List[str]:
    matches = re.findall(r"function\s+\w+\s*\([^)]*\)\s*(?:public|external|internal|private)?", code, flags=re.IGNORECASE)
    # 去重并保序
    seen = set()
    out = []
    for m in matches:
        s = re.sub(r"\s+", " ", m).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_compact_contract_context(code: str, max_chars: int = 2200) -> str:
    """Build concise context from full code for lower-token iterative prompts."""
    if not code:
        return ""

    signatures = _extract_function_signatures(code)
    sig_text = "\n".join(f"- {s}" for s in signatures[:30])

    # 保留头部结构信息 + 函数签名摘要
    compact_code = _compact_code_for_prompt(code, max_chars=max_chars)
    return (
        "=== CONTRACT SNAPSHOT (COMPACTED) ===\n"
        f"{compact_code}\n\n"
        "=== FUNCTION SIGNATURES ===\n"
        f"{sig_text}"
    )


def _extract_issue_type_from_feedback_text(text: str) -> str:
    """Extract issue type token from feedback text like '[HIGH] MISSING_GUARD: ...'."""
    s = str(text or "").strip()
    if not s:
        return "UNKNOWN"
    m = re.search(r"\b([A-Z_]{3,})\s*:", s)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def _derive_alignment_focus_items(priority_issues: List[Dict[str, str]]) -> List[str]:
    """Derive reusable generation focus items from issue types/descriptions.

    Notes:
    - Intentionally domain-agnostic (works across CUAD/MAUD and contract families).
    - Uses issue types as primary signal and description text as secondary hints.
    """
    if not priority_issues:
        return []

    type_focus_map = {
        "BEHAVIOR_SEMANTIC_MISMATCH": "Align transition semantics with legal intent: enforce the same trigger, guard, actor, and action behavior.",
        "MISSING_GUARD": "For each transition, keep state guard and add at least one legal-condition guard (time/role/compliance/notice).",
        "TIMING_CONSTRAINTS_MISSING": "Encode time windows/deadlines/holding periods as executable checks (e.g., block.timestamp constraints).",
        "MISSING_ACTION": "Implement concrete side effects (state updates, balance updates, and events) rather than marker-only transitions.",
        "MISSING_FUNCTIONALITY": "Add explicit executable workflow branches for missing legal obligations, including review/approval/execute steps when needed.",
        "AUTOMATIC_BEHAVIOR_MISSING": "Implement condition-driven enforcement functions for automatic/default legal behavior paths.",
        "DEFAULT_BEHAVIOR_MISMATCH": "Implement explicit default/no-action branch outcomes to match legal default semantics.",
        "INCOMPLETE_ACTORS": "Strengthen actor-role authorization checks to match legal actor responsibilities.",
        "MISSING_TRANSITION": "Add missing legal transition paths with complete guards and actions.",
        "INVALID_TRANSITION": "Block illegal state jumps and enforce legal transition order with strict preconditions.",
    }

    focus_items: List[str] = []
    seen_types = set()
    merged_text = "\n".join(
        f"{i.get('issue_type', '')} {i.get('description', '')} {i.get('suggestion', '')}"
        for i in priority_issues
    ).lower()

    for issue in priority_issues:
        issue_type = str(issue.get("issue_type", "UNKNOWN")).strip().upper()
        if issue_type in seen_types:
            continue
        seen_types.add(issue_type)
        mapped = type_focus_map.get(issue_type)
        if mapped:
            focus_items.append(mapped)

    # Secondary generic hints (still domain-agnostic)
    if any(k in merged_text for k in ["placeholder", "todo", "not implemented"]):
        focus_items.append("Replace placeholder/TODO logic with executable code paths that modify state and emit auditable events.")
    if any(k in merged_text for k in ["manual state", "state_only", "state-only"]):
        focus_items.append("Avoid unrestricted manual transition helpers; gate transition functions with legal predicates beyond state equality.")
    if any(k in merged_text for k in ["notice", "notify", "notification"]):
        focus_items.append("Persist notice status/timestamp and enforce downstream transitions based on notice conditions.")
    if any(k in merged_text for k in ["approval", "review", "authorize"]):
        focus_items.append("Model approval/review checkpoints explicitly and gate execution on approval outcome.")

    if not focus_items:
        focus_items.append("Resolve high-priority behavior mismatches by adding executable guards, actions, and explicit transition logic.")

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for item in focus_items:
        key = re.sub(r"\s+", " ", item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:8]


def _derive_action_anchors_from_constraints(
    constraints_chunk: List[Any],
    legal_transition_lines: Optional[List[str]] = None,
) -> List[str]:
    """Extract executable action anchors that should appear in code identifiers.

    Goal: reduce semantic-action misses by nudging model output to include explicit,
    machine-detectable action tokens in function/event/variable names.
    """
    anchor_patterns = {
        "pay": r"\b(pay|payment|compensation|remit|commission)\b",
        "transfer": r"\b(transfer|sale|sell|disposition)\b",
        "authorize": r"\b(authorize|authorisation|authorization|approve|approval|consent)\b",
        "notice": r"\b(notice|notify|notification)\b",
        "audit": r"\b(audit|inspection|books and records|records)\b",
        "terminate": r"\b(terminate|termination|expire|expired)\b",
        "breach": r"\b(breach|violation|violated|default)\b",
        "cure": r"\b(cure|cure period)\b",
        "indemnify": r"\b(indemnif|hold harmless|defend)\b",
        "confidentiality": r"\b(confidential|non-disclosure|nda|proprietary)\b",
        "renew": r"\b(renew|renewal|extend|extension)\b",
        "compliance": r"\b(compliance|comply|rule 145|registration|sec)\b",
        "lockup": r"\b(lockup|restrictive period|restriction period)\b",
        "legend_remove": r"\b(legend removed|remove legend|legend)\b",
        "consult_cooperate": r"\b(consult|cooperate|cooperation)\b",
        "filing_disclosure": r"\b(filing|disclose|disclosure|submit)\b",
        "legal_action": r"\b(litigation|injunction|legal action|final judgment)\b",
    }

    text_parts: List[str] = []
    for c_item in constraints_chunk or []:
        if isinstance(c_item, dict):
            text_parts.append(str(c_item.get("text", c_item.get("constraint", ""))))
        else:
            text_parts.append(str(c_item))
    if legal_transition_lines:
        text_parts.extend([str(x) for x in legal_transition_lines if x])

    merged = "\n".join(text_parts).lower()
    anchors: List[str] = []
    for anchor, pat in anchor_patterns.items():
        if re.search(pat, merged, re.IGNORECASE):
            anchors.append(anchor)

    # Keep compact and stable ordering for prompt reliability.
    return anchors[:10]


def _strip_comments(text: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", text or "", flags=re.DOTALL)
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def _extract_function_blocks(code: str) -> List[Dict[str, str]]:
    """Extract top-level function blocks using brace matching."""
    if not code:
        return []

    blocks: List[Dict[str, str]] = []
    pattern = re.compile(
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)[^{;]*\{",
        re.IGNORECASE,
    )
    for m in pattern.finditer(code):
        name = m.group(1)
        brace_start = code.find("{", m.start())
        if brace_start < 0:
            continue

        depth = 0
        end = -1
        for i in range(brace_start, len(code)):
            ch = code[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end < 0:
            continue

        blocks.append({
            "name": name,
            "signature": code[m.start():brace_start].strip(),
            "body": code[brace_start + 1:end - 1],
            "full": code[m.start():end],
        })

    return blocks


def _extract_require_conditions(body: str) -> List[str]:
    """Extract condition expressions from require(condition, ...)."""
    conditions: List[str] = []
    if not body:
        return conditions

    cursor = 0
    while cursor < len(body):
        m = re.search(r"\brequire\s*\(", body[cursor:], re.IGNORECASE)
        if not m:
            break

        start = cursor + m.end() - 1  # points to '('
        depth = 0
        end = -1
        for i in range(start, len(body)):
            ch = body[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end < 0:
            break

        raw_inside = body[start + 1:end].strip()
        cond = raw_inside.split(",", 1)[0].strip()
        if cond:
            conditions.append(cond)
        cursor = end + 1

    return conditions


def _is_state_only_require(condition: str) -> bool:
    c = re.sub(r"\s+", "", condition or "").strip("()")
    if re.fullmatch(r"currentState==ContractState\.[A-Za-z_][A-Za-z0-9_]*", c):
        return True
    if re.fullmatch(r"ContractState\.[A-Za-z_][A-Za-z0-9_]*==currentState", c):
        return True
    return False


def _detect_placeholder_implementations(code: str) -> List[Dict[str, str]]:
    """Detect high-risk placeholder stubs that should not pass to final output."""
    findings: List[Dict[str, str]] = []

    def _is_semantically_critical_function(name: str) -> bool:
        n = (name or "").strip()
        if not n:
            return False
        # Simple setter/getter stubs are common utility hooks and should not
        # block the full pipeline by default.
        if re.match(r"^(get|set)[A-Z_].*", n):
            return False
        return bool(
            re.search(
                r"enforce|check|verify|compliance|transfer|legend|rule|merger|sale|report|notice|risk|transition|^do[A-Z_]",
                n,
                re.IGNORECASE,
            )
        )

    for fn in _extract_function_blocks(code):
        name = fn.get("name", "unknown")
        body_raw = fn.get("body", "")
        body_wo_comments = _strip_comments(body_raw)
        body_compact = re.sub(r"\s+", " ", body_wo_comments).strip()
        is_critical = _is_semantically_critical_function(name)

        if not body_compact:
            if is_critical:
                findings.append({"function": name, "reason": "empty_or_comment_only_body"})
            continue

        if re.search(r"\bTODO\b|to be implemented|implementation for", body_raw, re.IGNORECASE):
            if is_critical:
                findings.append({"function": name, "reason": "todo_or_implementation_placeholder"})
            continue

        suspicious_name = is_critical or re.search(
            r"check|verify|compliance|allowed|rule|legend|restriction|enforce",
            name,
            re.IGNORECASE,
        )
        has_trivial_true = re.search(r"\breturn\s+true\s*;", body_compact, re.IGNORECASE)
        has_real_logic = re.search(
            r"\b(require|if|for|while)\b|block\.timestamp|&&|\|\||==|!=|>=|<=|>|<",
            body_compact,
            re.IGNORECASE,
        )
        if suspicious_name and has_trivial_true and not has_real_logic:
            findings.append({"function": name, "reason": "trivial_return_true_stub"})

    return findings


def _detect_transition_guard_gaps(code: str) -> List[str]:
    """Find transition functions that only guard by state but miss legal-condition guards."""
    missing: List[str] = []
    for fn in _extract_function_blocks(code):
        name = fn.get("name", "unknown")
        body = fn.get("body", "")

        # Transition-like function: explicitly changes FSM state.
        if not re.search(r"\bcurrentState\s*=\s*ContractState\.[A-Za-z_][A-Za-z0-9_]*", body):
            continue

        conditions = _extract_require_conditions(body)
        if not conditions:
            missing.append(f"{name}: missing_require_guards")
            continue

        has_non_state_require = any(not _is_state_only_require(c) for c in conditions)
        if not has_non_state_require:
            missing.append(f"{name}: state_only_guard")

    return missing

# =========================
# Agent2 主类（修复编译相关缺陷）
# =========================
class Agent2FSMToContract:
    """
    Agent2: 分阶段生成 Solidity 合约（业务约束 + 安全约束双落地，保证可编译）
    阶段1：生成可编译的合约骨架（含安全基础结构，无语法冲突）
    阶段2：填充业务约束 + 强制落地安全约束（优先保证编译正确性）
    阶段3：安全审计修复（移除高危语法，补充安全检查，兜底语法正确性）
    """

    def __init__(
            self,
            model_name: str,
            output_dir: str = "./Outputs/agent2_code",
            max_iterations: int = 5,
            include_security_rules: bool = False,
            strict_semantic_checks: bool = False,
            defer_compilation_fix: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.include_security_rules = include_security_rules
        self.strict_semantic_checks = strict_semantic_checks
        self.defer_compilation_fix = defer_compilation_fix
        self.active_security_rules = list(SECURITY_RULES) if include_security_rules else []
        self.cache_dir = os.path.join(self.output_dir, "_layered_cache")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_working_code_path(self, contract_id: str) -> str:
        """Working file path used during chunk-by-chunk updates."""
        safe_id = sanitize_filename(contract_id)
        return os.path.join(self.output_dir, f"{safe_id}_working.sol")

    def _fingerprint_payload(self, payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _get_layer_cache_path(self, contract_id: str, layer: str, cache_key: str) -> str:
        safe_id = sanitize_filename(contract_id)
        layer_dir = os.path.join(self.cache_dir, safe_id)
        os.makedirs(layer_dir, exist_ok=True)
        return os.path.join(layer_dir, f"{layer}_{cache_key}.json")

    def _load_cached_code(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            code = payload.get("code", "") if isinstance(payload, dict) else ""
            if isinstance(code, str) and _looks_like_valid_contract_code(code):
                return code
        except Exception as e:
            logger.debug(f"Load cache failed: {path}, error={e}")
        return ""

    def _save_cached_code(self, path: str, code: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(code, str) or not code.strip():
            return
        try:
            payload = {
                "code": code,
                "meta": meta or {},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Save cache failed: {path}, error={e}")

    def _write_working_code(self, path: str, code: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

    def _read_working_code(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # =========================
    # 主入口（改进：分离安全约束和业务约束处理）
    # =========================
    def run(
            self,
            fsm_result: Dict[str, Any],
            agent4_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        contract_id = fsm_result.get("contract_id", fsm_result.get("fsm_id", "unknown_contract"))
        logger.info(f"Agent2 start for contract_id={contract_id}")

        fsm = fsm_result.get("fsm", fsm_result)  # 兼容FSM直接作为顶层数据
        if not fsm:
            raise ValueError("FSM missing in Agent1 output")

        # 提取业务约束
        self.solidity_constraints = fsm_result.get("all_solidity_constraints", [])
        self.constraint_count = fsm_result.get("constraint_count", len(self.solidity_constraints))
        logger.info(f"Loaded {self.constraint_count} business constraints from Agent1")
        if self.include_security_rules:
            logger.info(f"Loaded {len(self.active_security_rules)} security constraints to enforce")
        else:
            logger.info("Security constraints disabled for this run")

        # 从结构化约束中提取合约角色（新格式 dicts 才有 actors 字段）
        roles = _extract_roles_from_constraints(self.solidity_constraints)[:6]
        if roles:
            logger.info(f"Identified {len(roles)} contract roles (capped at 6): {roles}")

        strict_semantic_checks = bool(fsm_result.get("strict_semantic_checks", self.strict_semantic_checks))
        logger.info(f"Semantic strict mode: {strict_semantic_checks}")

        # 阶段1：优先复用上一轮代码作为基线，避免每轮重生成导致回归
        previous_contract_code = fsm_result.get("previous_contract_code", "")
        legal_fsm_for_cache = fsm.get("legal_fsm", {}) if isinstance(fsm, dict) else {}
        skeleton_cache_key = self._fingerprint_payload({
            "layer": "stage1_skeleton",
            "contract_id": contract_id,
            "roles": roles,
            "legal_fsm": legal_fsm_for_cache,
        })
        skeleton_cache_path = self._get_layer_cache_path(contract_id, "stage1_skeleton", skeleton_cache_key)

        if isinstance(previous_contract_code, str) and _looks_like_valid_contract_code(previous_contract_code):
            skeleton_code = previous_contract_code
            logger.info("Stage 1 skipped: Reusing previous iteration contract as baseline")
        else:
            if previous_contract_code:
                logger.warning("Provided previous_contract_code is invalid, falling back to deterministic skeleton")
            cached_skeleton = self._load_cached_code(skeleton_cache_path)
            if cached_skeleton:
                skeleton_code = cached_skeleton
                logger.info("Stage 1 cache hit: Loaded cached deterministic skeleton")
            else:
                skeleton_code = self._generate_contract_skeleton(fsm, contract_id, roles=roles)
                self._save_cached_code(
                    skeleton_cache_path,
                    skeleton_code,
                    meta={"contract_id": contract_id, "layer": "stage1_skeleton", "roles": roles},
                )
                logger.info("Stage 1 completed: Generated contract skeleton")

        # 阶段2：填充业务约束（专注业务逻辑）
        stage2_cache_key = self._fingerprint_payload({
            "layer": "stage2_business",
            "contract_id": contract_id,
            "skeleton_hash": hashlib.md5(skeleton_code.encode("utf-8")).hexdigest(),
            "constraints": self.solidity_constraints,
            "agent4_feedback": agent4_feedback or {},
            "strict_semantic_checks": strict_semantic_checks,
        })
        stage2_cache_path = self._get_layer_cache_path(contract_id, "stage2_business", stage2_cache_key)
        cached_stage2 = self._load_cached_code(stage2_cache_path)
        if cached_stage2:
            code_with_business = cached_stage2
            logger.info("Stage 2 cache hit: Loaded cached business-filled contract")
        else:
            code_with_business = self._fill_business_constraints(
                skeleton_code,
                fsm,
                agent4_feedback,
                contract_id,
                roles=roles,
            )
            self._save_cached_code(
                stage2_cache_path,
                code_with_business,
                meta={"contract_id": contract_id, "layer": "stage2_business"},
            )
            logger.info("Stage 2 completed: Filled business constraints")

        final_code = code_with_business

        # 语义硬校验：禁止占位实现，且状态迁移必须包含法律条件guard
        final_code = self._repair_semantic_gaps(
            code=final_code,
            contract_id=contract_id,
            legal_constraints=self.solidity_constraints,
            agent4_feedback=agent4_feedback,
            max_passes=2,
            strict=strict_semantic_checks,
        )

        # 编译修复后置：在语义修复后统一执行，减少中途重复修复开销
        final_code = self._validate_and_fix_compilation(final_code, contract_id, stage="post_semantic")

        semantic_placeholder = _detect_placeholder_implementations(final_code)
        semantic_guard_gaps = _detect_transition_guard_gaps(final_code)
        if semantic_placeholder or semantic_guard_gaps:
            logger.warning(
                "Residual semantic findings after Agent2 run | "
                f"placeholder={len(semantic_placeholder)}, guard_gaps={len(semantic_guard_gaps)}"
            )

        # 统计落地情况
        business_implemented = self._count_implemented_constraints(final_code, self.solidity_constraints)

        # 构建最终结果
        final_result = {
            "contract_id": contract_id,
            "iteration": 1,
            "solidity_code": final_code,
            "fsm_assumptions": [],
            "agent4_feedback_used": agent4_feedback or {},
            "constraints_from_agent1": self.solidity_constraints,
            "business_constraints_implemented": business_implemented,
            "total_constraints_implemented": business_implemented,
            "contract_skeleton": skeleton_code,
            "semantic_findings": {
                "placeholder_functions": semantic_placeholder,
                "transition_guard_gaps": semantic_guard_gaps,
            },
        }

        self._save_output(contract_id, final_result)
        return final_result

    # =========================
    # 阶段1：生成骨架
    # =========================
    def _generate_contract_skeleton(self, fsm: Dict[str, Any], contract_id: str, roles: List[str] = None) -> str:
        """Build Stage 1 skeleton deterministically from Agent1's legal FSM data.

        改进：不包含安全约束，专注于业务结构，避免安全和业务逻辑混在一起。
        No model call is made here — using Agent1's extracted states and transitions
        guarantees correct enum and guard structure regardless of model behavior.
        The model will enrich the skeleton in Stage 2 when filling constraints.
        """
        contract_name = f"AffiliateContract_{sanitize_filename(contract_id)[:8]}"
        legal_fsm = fsm.get("legal_fsm", {})

        # 关键改进：骨架不包含安全约束，只包含业务结构
        skeleton_code = _build_skeleton_from_legal_fsm(
            contract_name, legal_fsm, roles or [], include_security_rules=False
        )

        if not _looks_like_valid_contract_code(skeleton_code):
            logger.warning("Deterministic skeleton invalid; falling back to basic skeleton")
            skeleton_code = _build_fallback_skeleton(contract_name, include_security_rules=False, roles=roles)

        logger.info(f"Generated contract skeleton: {len(skeleton_code)} chars, "
                    f"{skeleton_code.count('function ')} functions, "
                    f"{'enum ContractState' in skeleton_code}")
        return skeleton_code

    # =========================
    # 阶段2：填充业务约束
    # =========================
    def _fill_business_constraints(
            self,
            skeleton_code: str,
            fsm: Dict[str, Any],
            agent4_feedback: Optional[Dict[str, Any]] = None,
            contract_id: str = "unknown_contract",
            roles: List[str] = None,
    ) -> str:
        """分批填充业务约束，降低单次prompt压力，提升约束落地概率。

        改进：只处理业务约束，不混入安全约束，避免信息过载和逻辑冲突。
        """
        feedback_section = ""
        behavior_focus_section = ""
        focus_input_items: List[Dict[str, str]] = []
        if agent4_feedback and isinstance(agent4_feedback, dict):
            structured_issues = agent4_feedback.get("structured_issues", [])
            if isinstance(structured_issues, list) and structured_issues:
                severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                sorted_structured = sorted(
                    [i for i in structured_issues if isinstance(i, dict)],
                    key=lambda x: severity_rank.get(x.get("severity", "low"), 1),
                    reverse=True
                )
                top_structured = sorted_structured[:12]
                issue_type_counts: Dict[str, int] = {}
                formatted_items = []
                for item in top_structured:
                    issue_type = str(item.get("issue_type", "UNKNOWN")).strip().upper() or "UNKNOWN"
                    severity = str(item.get("severity", "medium")).strip().lower() or "medium"
                    location = str(item.get("location", "unknown"))
                    description = str(item.get("description", ""))
                    suggestion = str(item.get("suggestion", ""))
                    issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
                    formatted_items.append(
                        f"- [{severity.upper()}][{issue_type}] {location}: {description} | Fix: {suggestion}"
                    )
                    focus_input_items.append({
                        "issue_type": issue_type,
                        "severity": severity,
                        "description": description,
                        "location": location,
                        "suggestion": suggestion,
                    })

                type_summary = ", ".join(f"{k}={v}" for k, v in sorted(issue_type_counts.items(), key=lambda kv: (-kv[1], kv[0])))
                feedback_section = (
                    "\n======================\n"
                    "AGENT4 STRUCTURED FEEDBACK (previous iteration issues)\n"
                    "======================\n"
                    f"Issue Type Summary: {type_summary}\n"
                    + "\n".join(formatted_items)
                )
            else:
                issues = agent4_feedback.get("issues")
                if issues:
                    unique_issues = []
                    seen_issues = set()
                    for item in issues:
                        text = str(item).strip()
                        if not text:
                            continue
                        key = re.sub(r"\s+", " ", text).lower()
                        if key in seen_issues:
                            continue
                        seen_issues.add(key)
                        unique_issues.append(text)
                        issue_type = _extract_issue_type_from_feedback_text(text)
                        sev = "medium"
                        lower_text = text.lower()
                        if "critical" in lower_text:
                            sev = "critical"
                        elif "high" in lower_text:
                            sev = "high"
                        elif "low" in lower_text:
                            sev = "low"
                        focus_input_items.append({
                            "issue_type": issue_type,
                            "severity": sev,
                            "description": text,
                            "location": "unknown",
                            "suggestion": "",
                        })
                        if len(unique_issues) >= 12:
                            break
                    if unique_issues:
                        feedback_section = (
                            "\n======================\nAGENT4 FEEDBACK (previous iteration issues)\n======================\n"
                            + "\n".join(f"- {i}" for i in unique_issues)
                        )

            focus_items = _derive_alignment_focus_items(focus_input_items)
            if focus_items:
                behavior_focus_section = (
                    "\n======================\n"
                    "BEHAVIOR-SEMANTIC FIX PRIORITY\n"
                    "======================\n"
                    + "\n".join(f"{idx}. {item}" for idx, item in enumerate(focus_items, start=1))
                )

        logger.debug(
            f"Business constraints being sent to model ({self.constraint_count}); sample={self.solidity_constraints[:3]}"
        )

        # 分批约束：减小批次大小以降低信息过载，提高约束落地准确率
        # 分离链上/链外约束，只对链上约束分批调用模型
        on_chain_constraints = []
        off_chain_constraints = []
        constraint_by_type = {
            "state_transition": [],
            "access_control": [],
            "timing_constraint": [],
            "role_obligation": [],
            "other": []
        }

        for c in self.solidity_constraints:
            if isinstance(c, dict):
                text = c.get("constraint", "")
                enforcement = c.get("enforcement", "on_chain")
                constraint_type = c.get("type", "other")
                actors = c.get("actors") or []
            else:
                text = str(c)
                enforcement = "on_chain"
                constraint_type = "other"
                actors = []

            if not text:
                continue

            item = {
                "text": text,
                "type": constraint_type,
                "actors": actors if isinstance(actors, list) else [],
                "enforcement": enforcement,
            }

            if enforcement == "off_chain":
                off_chain_constraints.append(item)
            else:
                on_chain_constraints.append(item)
                # 按类型分组
                if constraint_type in constraint_by_type:
                    constraint_by_type[constraint_type].append(item)
                else:
                    constraint_by_type["other"].append(item)

        logger.info(f"Constraint split: {len(on_chain_constraints)} on-chain, {len(off_chain_constraints)} off-chain")
        logger.info(f"Constraint types: state_transition={len(constraint_by_type['state_transition'])}, "
                   f"access_control={len(constraint_by_type['access_control'])}, "
                   f"timing={len(constraint_by_type['timing_constraint'])}, "
                   f"role={len(constraint_by_type['role_obligation'])}, "
                   f"other={len(constraint_by_type['other'])}")

        # 关键改进：动态调整chunk_size，适应不同规模的合约，最大为5确保细节不丢失
        total_constraints = len(on_chain_constraints)
        if total_constraints <= 10:
            chunk_size = 4  # 小合约
        else:
            chunk_size = 5  # 高质量细节模式

        logger.info(f"Using chunk_size={chunk_size} for {total_constraints} constraints")

        # 优先级排序：state_transition > access_control > timing_constraint > role_obligation > other
        prioritized_constraints = []
        for priority_type in ["state_transition", "access_control", "timing_constraint", "role_obligation", "other"]:
            prioritized_constraints.extend(constraint_by_type[priority_type])

        chunks = [prioritized_constraints[i:i + chunk_size] for i in range(0, len(prioritized_constraints), chunk_size)]
        current_code = skeleton_code

        # Extract state labels from Agent1 FSM for repair prompt
        legal_fsm_data = fsm.get("legal_fsm", {})
        repair_state_labels = [s.get("label") for s in legal_fsm_data.get("states", []) if s.get("label")][:12]
        repair_state_enum_str = ", ".join(repair_state_labels) if repair_state_labels else "STATE_0, STATE_1, STATE_2"
        contract_name = f"AffiliateContract_{sanitize_filename(contract_id)[:8]}"

        # 文件驱动：每轮从本地工作文件读取最新代码，而不是仅依赖内存变量
        working_code_path = self._get_working_code_path(contract_id)
        self._write_working_code(working_code_path, current_code)

        # 长合约处理：增量验证和早停机制
        failed_chunks = 0  # 连续失败的批次数
        total_chunks = len(chunks)
        verification_interval = max(10 // chunk_size, 3)  # 每处理约10个约束验证一次

        for idx, chunk in enumerate(chunks, start=1):
            # 关键改进：每批都附带agent4反馈，让模型持续看到需要修复的问题
            chunk_feedback_section = feedback_section
            if behavior_focus_section:
                chunk_feedback_section += "\n" + behavior_focus_section
            current_code = self._read_working_code(working_code_path) or current_code
            
            # 避免第三方中转网关在发送超长请求时引发 ProxyError，单独对 deepseek 限制上下文
            target_max_chars = 4500 if "deepseek" in self.model_name.lower() else 8000
            prompt_context = _build_compact_contract_context(current_code, max_chars=target_max_chars)

            unresolved_placeholders = _detect_placeholder_implementations(current_code)
            unresolved_guard_gaps = _detect_transition_guard_gaps(current_code)
            remediation_section = ""
            if unresolved_placeholders or unresolved_guard_gaps:
                placeholder_lines = [
                    f"- {x.get('function', 'unknown')}: {x.get('reason', 'unknown')}"
                    for x in unresolved_placeholders[:12]
                ]
                guard_lines = [f"- {g}" for g in unresolved_guard_gaps[:12]]
                remediation_section = (
                    "\n======================\n"
                    "UNRESOLVED IMPLEMENTATION GAPS (MUST FIX IN THIS PASS)\n"
                    "======================\n"
                    "Placeholder functions:\n"
                    f"{chr(10).join(placeholder_lines) if placeholder_lines else '- none'}\n\n"
                    "State-transition guard gaps:\n"
                    f"{chr(10).join(guard_lines) if guard_lines else '- none'}\n"
                )

            # 进度报告（对于长合约）
            if total_chunks > 10:
                logger.info(f"Processing constraints: batch {idx}/{total_chunks} ({idx*chunk_size}/{total_constraints} constraints)")

            # 提取Legal FSM上下文，用于语义对齐
            legal_fsm_data = fsm.get("legal_fsm", {})
            legal_states = [s.get("label", s.get("name", "")) for s in legal_fsm_data.get("states", [])]
            legal_transitions = legal_fsm_data.get("transitions", [])
            state_id_to_label = {}
            for s in legal_fsm_data.get("states", []):
                sid = s.get("state_id")
                lbl = s.get("label", s.get("name", ""))
                if sid and lbl:
                    state_id_to_label[str(sid)] = str(lbl)

            # 格式化法律状态机的转换摘要
            legal_transitions_summary = []
            for t in legal_transitions[:10]:
                from_state_raw = str(t.get("from", ""))
                to_state_raw = str(t.get("to", ""))
                from_state = state_id_to_label.get(from_state_raw, from_state_raw)
                to_state = state_id_to_label.get(to_state_raw, to_state_raw)
                trigger = str(t.get("trigger", t.get("action", "")))
                guards = t.get("guards", [])
                if isinstance(guards, str):
                    guards = [guards]
                if not guards:
                    single_guard = t.get("guard", "")
                    if single_guard:
                        guards = [str(single_guard)]
                if from_state and to_state:
                    guard_str = f" (guards: {', '.join(guards[:2])})" if guards else ""
                    legal_transitions_summary.append(f"{from_state} → {to_state} [trigger: {trigger}]{guard_str}")

            legal_states_str = ", ".join(legal_states) if legal_states else "Not specified"
            legal_transitions_str = "\n".join(legal_transitions_summary) if legal_transitions_summary else "Not specified"

            # 格式化约束，添加类型标签和上下文
            formatted_constraints = []
            for i, c_item in enumerate(chunk, 1):
                if isinstance(c_item, dict):
                    text = c_item.get("text", c_item.get("constraint", ""))
                    constraint_type = c_item.get("type", "other")
                    actors = c_item.get("actors") or []
                    type_label = f"[{constraint_type}]"
                else:
                    text = str(c_item)
                    actors = []
                    type_label = "[other]"
                actor_suffix = f" | actors: {', '.join([str(a) for a in actors[:3]])}" if actors else ""
                formatted_constraints.append(f"{i}. {type_label} {text}{actor_suffix}")

            formatted_constraints_str = "\n".join(formatted_constraints)
            action_anchors = _derive_action_anchors_from_constraints(
                constraints_chunk=chunk,
                legal_transition_lines=legal_transitions_summary,
            )
            action_anchors_str = ", ".join(action_anchors) if action_anchors else "none"

            # 改进：强调法律行为语义与智能合约代码行为语义的对齐
            constraint_prompt = f"""
You are a legal-tech Solidity engineer specializing in translating legal behavior into smart contract code.

YOUR PRIMARY GOAL: Ensure the smart contract's BEHAVIOR semantically aligns with the legal FSM's BEHAVIOR.
This means: when the legal FSM defines a state transition with specific conditions and actions,
the smart contract code must enforce the SAME transition logic, conditions, and actions.

======================
LEGAL FSM CONTEXT (Target Behavior)
======================
States: {legal_states_str}
Key Transitions:
{legal_transitions_str}

This is the INTENDED behavior from the legal document. Your code must implement this behavior faithfully.

======================
CURRENT CODE CONTEXT
======================
{prompt_context}

======================
BUSINESS CONSTRAINTS TO IMPLEMENT (Batch {idx}/{total_chunks})
======================
{formatted_constraints_str}

======================
ACTION ANCHORS (MUST APPEAR IN EXECUTABLE IDENTIFIERS)
======================
{action_anchors_str}
Use these anchor tokens in function/event/variable names when implementing related constraints.
Examples: pay -> processPay, notice -> giveNotice, audit -> performAudit.
Do not put anchors only in comments.

======================
SEMANTIC ALIGNMENT REQUIREMENTS
======================
For EACH constraint above:
1. UNDERSTAND the legal intent: What behavior is the legal document requiring?
2. IMPLEMENT the behavior: Write Solidity code that enforces this exact behavior
3. VERIFY semantic alignment: Does your code's behavior match the legal requirement?
4. ADD implementation comment: // Constraint [N]: <brief explanation of how code implements legal intent>

CRITICAL RULES:
1. Semantic correctness FIRST: Code behavior must match legal behavior
2. Then ensure compilability: Code must compile with Solidity {SOLIDITY_VERSION}
3. PRESERVE FSM structure: DO NOT remove/rename enum ContractState, currentState, or state guards
4. Only ADD logic; do not delete existing implemented constraints
5. STRICT MINIMALISM: You MUST STRICTLY implement ONLY the states, actions, and transitions defined in the provided FSM. DO NOT add unrelated public/external behaviors.
6. NO HALLUCINATION: DO NOT add any boilerplate modifiers (like Ownable, Pausable), access controls not explicitly specified, fallback functions, or extra safety triggers unless explicitly ordered by the FSM. Only code what is strictly necessary.
7. If UNRESOLVED IMPLEMENTATION GAPS are listed below, you MUST replace those placeholders with executable logic in this pass.
8. Keep comments minimal and short: only allow brief "// Constraint [N]: ..." implementation comments; do NOT include long explanatory prose comments.
9. If an action anchor is relevant to a constraint, include that anchor in executable identifiers (function/event/variable), not only comments.

======================
CRITICAL: AUTOMATIC BEHAVIORS
======================
When legal text says "automatically" or "shall renew unless", the behavior is MANDATORY, not optional.
DO NOT implement automatic behaviors as functions that require explicit calls by specific parties.
INSTEAD, implement them as enforcement functions that ANY party can call to trigger the automatic logic.

Example - Auto-renewal clause:
Legal: "The Agreement shall automatically renew unless either party gives 60 days' notice"

❌ WRONG (requires explicit call, renewal is optional):
function renewTerm() external {{
    if (block.timestamp >= termEnd - NOTICE_PERIOD) {{
        termEnd += RENEWAL_TERM;
    }}
}}
// Problem: If no one calls renewTerm(), nothing happens. But legally, renewal is AUTOMATIC!

✅ CORRECT (automatic behavior based on conditions):
bool public noticeGiven;

function giveNotice() external {{
    require(msg.sender == partyA || msg.sender == partyB, "Unauthorized");
    require(block.timestamp <= termEnd - NOTICE_PERIOD, "Too late");
    noticeGiven = true;
}}

function enforceTerm() external {{  // Any party can call this
    if (block.timestamp >= termEnd) {{
        if (!noticeGiven) {{
            termEnd += RENEWAL_TERM;  // Automatic renewal
            noticeGiven = false;
        }} else {{
            terminated = true;  // Termination
        }}
    }}
}}
// Correct: Renewal happens automatically when conditions are met (time + no notice)

======================
EXAMPLES OF SEMANTIC ALIGNMENT
======================
GOOD: Legal says "shares are locked for 6 months after merger"
→ Code: require(block.timestamp >= mergerDate + 180 days, "Lock-up period not expired");

BAD: Legal says "shares are locked for 6 months after merger"
→ Code: require(sharesLocked == false, "Shares locked"); // Missing time check!

GOOD: Legal says "only the acquirer can initiate merger"
→ Code: require(msg.sender == acquirer, "Only acquirer can initiate merger");

BAD: Legal says "only the acquirer can initiate merger"
→ Code: require(msg.sender != address(0), "Invalid sender"); // Too weak!

GOOD: Legal says "transition from PENDING to ACTIVE requires approval"
→ Code: require(currentState == ContractState.PENDING && isApproved, "Not approved");
       currentState = ContractState.ACTIVE;

BAD: Legal says "transition from PENDING to ACTIVE requires approval"
→ Code: currentState = ContractState.ACTIVE; // Missing state guard and approval check!
{remediation_section}
{chunk_feedback_section}

Output ONLY the updated Solidity code (no markdown, no explanations outside code comments):
"""

            logger.debug(f"Constraint prompt chunk {idx} (first 1200 chars):\n{constraint_prompt[:1200]}")

            # 关键改进：增加max_tokens从2800到4000，给模型更多空间生成完整代码
            resp = call_model(
                model_name=self.model_name,
                prompt=constraint_prompt,
                temperature=0.0,
                max_tokens=4000
            )
            next_code = _extract_and_clean_solidity_code(resp)
            if not _looks_like_valid_contract_code(next_code):
                logger.warning(f"Stage 2 chunk {idx} output invalid; keep previous code")
                failed_chunks += 1
                # 早停机制：连续3批失败，提前终止
                if failed_chunks >= 3:
                    logger.error(f"Stage 2 early stop: {failed_chunks} consecutive chunks failed")
                    break
                continue

            # 如果模型没改动（常见退化），记录并继续下一批
            if next_code.strip() == current_code.strip():
                logger.debug(f"Stage 2 chunk {idx}: model returned unchanged code")
                failed_chunks += 1
            else:
                failed_chunks = 0  # 重置失败计数

            current_code = next_code
            self._write_working_code(working_code_path, current_code)

            # 可选增量编译修复：默认关闭并后置到最终阶段，减少额外耗时
            if (not self.defer_compilation_fix) and total_chunks > 10 and idx % verification_interval == 0:
                logger.info(f"Incremental validation at batch {idx}/{total_chunks}")
                compiler = get_compiler()
                if compiler.is_available:
                    validation = compiler.validate_and_suggest_fixes(current_code, contract_id)
                    if not validation["success"]:
                        logger.warning(f"Incremental validation failed with {len(validation['errors'])} errors")
                        # 尝试修复编译错误
                        current_code = self._validate_and_fix_compilation(current_code, contract_id, stage=f"incremental_{idx}")
                        self._write_working_code(working_code_path, current_code)
                    else:
                        logger.info(f"Incremental validation passed")

        # 结构门槛校验：不达标时做一次“结构修复”重试（不注入法律语义）
        if not _meets_min_fsm_shape(current_code):
            metrics = _extract_structure_metrics(current_code)
            logger.warning(f"Stage 2 output lacks minimal FSM shape: {metrics}; attempting one structure-repair pass")
            repair_prompt = f"""
You are a Solidity engineer. Repair the contract to restore a parseable finite-state-machine structure.
Use EXACTLY these state names from the legal FSM: {repair_state_enum_str}
Keep all existing business logic text and comments. Only restore missing FSM structure.

Required structural properties:
1. At least one `enum` for states.
2. A public state variable (e.g., currentState) initialized in constructor.
3. At least two transition functions with both:
    - a guard `require(currentState == ContractState.X, ...)`
    - a transition assignment `currentState = ContractState.Y;`

Current code:
{current_code}

Return ONLY compilable Solidity code.
"""
            repaired = _extract_and_clean_solidity_code(
                call_model(
                    model_name=self.model_name,
                    prompt=repair_prompt,
                    temperature=0.0,
                    max_tokens=3000
                )
            )
            if _looks_like_valid_contract_code(repaired) and _meets_min_fsm_shape(repaired):
                current_code = repaired
                self._write_working_code(working_code_path, current_code)
                logger.info("Stage 2 structure-repair succeeded")
            else:
                logger.warning("Stage 2 structure-repair failed; falling back to generic FSM scaffold")
                # Use legal FSM scaffold if available, else generic (without security constraints)
                fallback = _build_skeleton_from_legal_fsm(
                    contract_name, legal_fsm_data, roles or [], include_security_rules=False
                ) if legal_fsm_data.get("states") else _build_fallback_fsm_scaffold(contract_name)
                current_code = fallback if _looks_like_valid_contract_code(fallback) else _build_fallback_fsm_scaffold(contract_name)
                self._write_working_code(working_code_path, current_code)

        if not _looks_like_valid_contract_code(current_code):
            logger.warning("Stage 2 output remained invalid; falling back to skeleton code")
            return skeleton_code

        return current_code

    def _repair_semantic_gaps(
            self,
            code: str,
            contract_id: str,
            legal_constraints: List[Any],
            agent4_feedback: Optional[Dict[str, Any]] = None,
            max_passes: int = 2,
            strict: bool = False,
    ) -> str:
        """Repair semantic stubs and enforce non-trivial legal guards on transitions.

        Hard-fail behavior: if unresolved after max passes, raise ValueError so
        placeholders cannot silently pass into final artifacts.
        """
        current_code = code

        for pass_idx in range(1, max_passes + 1):
            placeholder_findings = _detect_placeholder_implementations(current_code)
            guard_gaps = _detect_transition_guard_gaps(current_code)

            if not placeholder_findings and not guard_gaps:
                if pass_idx > 1:
                    logger.info("Semantic repair succeeded: no placeholder or guard-gap findings remain")
                return current_code

            logger.warning(
                f"Semantic findings before repair pass {pass_idx}/{max_passes}: "
                f"placeholders={len(placeholder_findings)}, transition_guard_gaps={len(guard_gaps)}"
            )

            placeholder_lines = [
                f"- {x.get('function', 'unknown')}: {x.get('reason', 'unknown')}"
                for x in placeholder_findings[:16]
            ]
            guard_lines = [f"- {g}" for g in guard_gaps[:16]]

            constraint_lines: List[str] = []
            for i, c in enumerate(legal_constraints[:16], start=1):
                if isinstance(c, dict):
                    c_text = str(c.get("constraint", "")).strip()
                    c_type = str(c.get("type", "other")).strip()
                    actors = c.get("actors") or []
                    actor_text = ", ".join([str(a) for a in actors[:3]]) if isinstance(actors, list) else ""
                    constraint_lines.append(f"{i}. [{c_type}] {c_text} | actors: {actor_text}")
                else:
                    constraint_lines.append(f"{i}. {str(c)}")

            feedback_lines: List[str] = []
            if isinstance(agent4_feedback, dict):
                structured = agent4_feedback.get("structured_issues", [])
                if isinstance(structured, list) and structured:
                    for item in structured[:8]:
                        if isinstance(item, dict):
                            issue_type = item.get("issue_type", "UNKNOWN")
                            desc = item.get("description", "")
                            feedback_lines.append(f"- [{issue_type}] {desc}")
                else:
                    issues = agent4_feedback.get("issues", [])
                    if isinstance(issues, list):
                        feedback_lines.extend([f"- {str(i)}" for i in issues[:8]])

            repair_prompt = f"""
You are a Solidity semantic-repair engineer.

TASK: Repair the contract so that legal behavior is executable, not placeholder text.

Detected placeholder findings:
{chr(10).join(placeholder_lines) if placeholder_lines else '- none'}

Detected transition guard gaps (state-only transitions):
{chr(10).join(guard_lines) if guard_lines else '- none'}

Legal constraints to ground the fixes:
{chr(10).join(constraint_lines) if constraint_lines else '- none'}

Agent4 behavior-semantic feedback (if any):
{chr(10).join(feedback_lines) if feedback_lines else '- none'}

MANDATORY REPAIR RULES:
1. Replace placeholder function bodies with executable logic.
2. For every transition function that sets currentState = ContractState.X, keep a state require guard AND add at least one legal-condition require guard (time/role/compliance/notice).
3. Do NOT keep trivial stubs like "return true;" in check/verify/compliance functions.
4. Preserve existing contract name, state enum names, public variables, and function signatures.
5. Keep Solidity version {SOLIDITY_VERSION} and return full compilable contract.
6. STRICT MINIMALISM & NO HALLUCINATION: Address the detected gaps only. DO NOT add new unrelated functions, modifiers (like Ownable), or extraneous safety triggers not mandated by the legal constraints. Do not over-engineer.

Current contract code:
{current_code}

Output ONLY the full repaired Solidity code.
"""

            resp = call_model(
                model_name=self.model_name,
                prompt=repair_prompt,
                temperature=0.0,
                max_tokens=4000,
            )
            candidate = _extract_and_clean_solidity_code(resp)
            if not _looks_like_valid_contract_code(candidate):
                logger.warning(f"Semantic repair pass {pass_idx}: model returned invalid contract; keeping previous")
                continue

            current_code = candidate

        # Hard-fail only in strict mode; otherwise warn and continue.
        final_placeholder = _detect_placeholder_implementations(current_code)
        final_guard_gaps = _detect_transition_guard_gaps(current_code)
        if final_placeholder or final_guard_gaps:
            sample_placeholder = ", ".join(
                f"{x.get('function')}:{x.get('reason')}" for x in final_placeholder[:5]
            ) or "none"
            sample_gaps = ", ".join(final_guard_gaps[:5]) or "none"
            if strict:
                logger.warning(
                    f"Semantic hard checks failed after repair passes. Proceeding anyway. | "
                    f"placeholder={len(final_placeholder)} ({sample_placeholder}) | "
                    f"guard_gaps={len(final_guard_gaps)} ({sample_gaps})"
                )
            logger.warning(
                "Semantic checks not fully resolved (non-strict mode), continue pipeline | "
                f"placeholder={len(final_placeholder)} ({sample_placeholder}) | "
                f"guard_gaps={len(final_guard_gaps)} ({sample_gaps})"
            )

        return current_code

    # =========================
    # 编译验证与修复（新增：确保代码可编译）
    # =========================
    def _validate_and_fix_compilation(self, code: str, contract_id: str, stage: str = "unknown") -> str:
        """验证代码是否可编译，如果失败则尝试修复

        Args:
            code: Solidity代码
            contract_id: 合约ID
            stage: 当前阶段（用于日志）

        Returns:
            修复后的代码（如果需要修复）或原始代码
        """
        compiler = get_compiler()

        # 如果编译器不可用，跳过验证
        if not compiler.is_available:
            logger.info(f"[{stage}] Solidity compiler not available, skipping validation")
            return code

        logger.info(f"[{stage}] Validating compilation for contract {contract_id}")
        validation_result = compiler.validate_and_suggest_fixes(code, contract_id)

        if validation_result["success"]:
            logger.info(f"[{stage}] Compilation successful")
            return code

        # 编译失败，尝试修复
        errors = validation_result["errors"]
        suggestions = validation_result["suggestions"]

        logger.warning(f"[{stage}] Compilation failed with {len(errors)} errors")
        logger.info(f"[{stage}] Attempting to fix compilation errors")

        # 构建修复prompt
        fix_prompt = f"""
You are a Solidity expert. Fix the compilation errors in the following contract.

COMPILATION ERRORS:
{chr(10).join([f"- {e}" for e in errors[:10]])}

SUGGESTED FIXES:
{chr(10).join([f"- {s}" for s in suggestions]) if suggestions else "No specific suggestions"}

CRITICAL RULES:
1. Fix ONLY the compilation errors, do not change business logic
2. Preserve all existing function signatures and state variables
3. Ensure the code is 100% compilable with Solidity {SOLIDITY_VERSION}
4. Do not remove or rename any states, transitions, or core logic

CURRENT CODE:
{code}

Output ONLY the fixed compilable Solidity code (no markdown, no explanations):
"""

        try:
            resp = call_model(
                model_name=self.model_name,
                prompt=fix_prompt,
                temperature=0.0,
                max_tokens=4000
            )
            fixed_code = _extract_and_clean_solidity_code(resp)

            # 验证修复后的代码
            if _looks_like_valid_contract_code(fixed_code):
                revalidation = compiler.validate_and_suggest_fixes(fixed_code, contract_id)
                if revalidation["success"]:
                    logger.info(f"[{stage}] Compilation errors fixed successfully")
                    return fixed_code
                else:
                    logger.warning(f"[{stage}] Fixed code still has {len(revalidation['errors'])} errors, keeping original")
                    return code
            else:
                logger.warning(f"[{stage}] Fixed code invalid, keeping original")
                return code

        except Exception as e:
            logger.error(f"[{stage}] Failed to fix compilation errors: {e}")
            return code

    # =========================
    # 工具函数：通用约束统计（支持业务+安全）
    # =========================
    def _count_implemented_constraints(self, solidity_code: str, constraints: List[str]) -> int:
        """统计已实现的约束数量（支持业务/安全约束，兼容字符串和字典格式）"""
        if not solidity_code or not constraints:
            return 0

        implemented = 0
        code_lower = solidity_code.lower()
        for constraint in constraints:
            text = constraint.get("constraint", "") if isinstance(constraint, dict) else str(constraint)
            keywords = re.findall(r"[a-zA-Z0-9_./()%-]+", text.lower())
            keywords = [k for k in keywords if len(k) > 3 and k not in {"then", "with", "that", "from", "under", "where", "after", "before", "during"}]
            if len(keywords) == 0:
                continue
            match_count = sum(1 for k in keywords if k in code_lower)
            match_ratio = match_count / len(keywords)
            # 旧规则(>=70%)对自然语言约束过于严格，几乎总是0。
            # 新规则：命中率>=30% 或者命中至少2个关键词视为“部分实现”。
            if match_ratio >= 0.3 or match_count >= 2:
                implemented += 1
            else:
                # log missing constraint for diagnostics
                logger.debug(f"Constraint not implemented or matched poorly: '{text}' keywords={keywords} match_count={match_count}")
        logger.debug(f"Total implemented constraints: {implemented}/{len(constraints)}")
        return implemented

    # =========================
    # 工具函数：保存输出（保留）
    # =========================
    def _save_output(self, contract_id: str, result: Dict[str, Any]):
        sanitized_contract_id = sanitize_filename(contract_id)
        out_path = os.path.join(
            self.output_dir,
            f"{sanitized_contract_id}_secure_iter{result['iteration']}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if self.include_security_rules:
                logger.info(
                    f"Agent2 output saved | Business constraints: {result['business_constraints_implemented']}/{self.constraint_count} | "
                    f"Security constraints: {result['security_constraints_implemented']}/{len(self.active_security_rules)}"
                )
            else:
                logger.info(
                    f"Agent2 output saved | Business constraints: {result['business_constraints_implemented']}/{self.constraint_count}"
                )
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            raise