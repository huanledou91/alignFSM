import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from llm_router import call_model  # 与 Agent1/Agent2 保持统一接口

# =========================
# 日志配置
# =========================
logger = logging.getLogger("Agent4")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - Agent4 - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

# =========================
# 问题类型和严重程度定义
# =========================
ISSUE_TYPES = {
    "MISSING_STATE": "Contract FSM缺少Legal FSM中定义的状态",
    "EXTRA_STATE": "Contract FSM包含Legal FSM中未定义的状态",
    "MISSING_TRANSITION": "Contract FSM缺少Legal FSM中定义的转换",
    "INVALID_TRANSITION": "Contract FSM包含不符合Legal FSM的转换",
    "MISSING_GUARD": "转换缺少必要的守卫条件",
    "MISSING_ACTION": "转换缺少必要的动作",
    "MISSING_FUNCTIONALITY": "缺少重要的业务功能",
    "INCOMPLETE_ACTORS": "角色/参与者定义不完整",
    "TIMING_CONSTRAINTS_MISSING": "缺少时间约束",
    "BEHAVIOR_SEMANTIC_MISMATCH": "代码行为语义与法律行为语义不一致",
    "AUTOMATIC_BEHAVIOR_MISSING": "法律要求的自动行为被实现为需要显式调用",
    "DEFAULT_BEHAVIOR_MISMATCH": "默认行为不匹配（法律默认行为与代码默认行为不一致）",
}

SEVERITY_LEVELS = {
    "critical": 4,  # 核心功能缺失，合约无法正常工作
    "high": 3,      # 重要功能缺失，影响主要业务逻辑
    "medium": 2,    # 次要功能缺失，影响部分业务逻辑
    "low": 1        # 细节问题，不影响核心功能
}

CATEGORY_TYPES = ["state", "transition", "guard", "action", "functionality", "actor", "timing", "behavior"]

# =========================
# Agent4 主类
# =========================
class Agent4FSMAnalyzer:
    """
    Agent4:
    - 输入: Legal FSM (Agent1 输出) 和 Contract FSM (Agent3 输出)
    - 对比两个 FSM 差异，生成 Agent2 可用的修复反馈
    """

    def __init__(self, model_name: str, output_dir: str = "./Outputs/agent4_feedback"):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================
    # 主入口（改进：生成结构化反馈）
    # =========================
    def run(
        self,
        contract_id: str,
        legal_fsm: Dict[str, Any],
        contract_fsm: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Agent4 start analysis for contract_id={contract_id}")

        # 构建分析 Prompt（改进：要求结构化输出）
        prompt = self._build_structured_analysis_prompt(legal_fsm, contract_fsm)

        # 调用 LLM 生成差异分析
        response = call_model(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=3000,  # 增加token限制以支持更详细的反馈
            # 保留原始文本，由 Agent4 自身完成数组/对象解析，避免路由层裁剪破坏 JSON 数组
            expect_json=False
        )

        # 核心修复：提取 response 中的字符串内容（兼容 dict/str 两种返回格式）
        llm_output = response.get("text", "") if isinstance(response, dict) else str(response)

        # 解析反馈为结构化JSON对象列表
        llm_issues = self._parse_structured_feedback(llm_output)

        # 规则检测：补充LLM可能漏掉的行为语义问题
        rule_issues = self._detect_rule_based_semantic_issues(legal_fsm, contract_fsm)

        # 合并并去重
        structured_issues = self._merge_dedup_issues(llm_issues, rule_issues)

        # 智能过滤：移除可能的误报
        filtered_issues = self._filter_false_positives(structured_issues, legal_fsm, contract_fsm)

        # 按严重程度排序
        sorted_issues = self._sort_by_severity(filtered_issues)

        # 生成统计信息
        stats = self._generate_statistics(sorted_issues)

        # 生成简化的issues列表（向后兼容）
        simple_issues = [self._format_issue_as_string(issue) for issue in sorted_issues]

        feedback_result = {
            "contract_id": contract_id,
            "issues": simple_issues,  # 向后兼容：字符串列表
            "structured_issues": sorted_issues,  # 新增：结构化问题列表
            "statistics": stats,  # 新增：统计信息
            "raw_llm_output": response
        }

        # 保存本地 JSON
        self._save_feedback(contract_id, feedback_result)

        logger.info(f"Agent4 feedback generated for contract_id={contract_id}")
        logger.info(f"  Total issues: {len(sorted_issues)}")
        logger.info(f"  By severity: Critical={stats['by_severity']['critical']}, "
                   f"High={stats['by_severity']['high']}, "
                   f"Medium={stats['by_severity']['medium']}, "
                   f"Low={stats['by_severity']['low']}")
        logger.info(f"  By category: {', '.join([f'{k}={v}' for k, v in stats['by_category'].items()])}")

        return feedback_result

    # =========================
    # 构建结构化差异分析 Prompt（改进）
    # =========================
    def _build_structured_analysis_prompt(self, legal_fsm: Dict[str, Any], contract_fsm: Dict[str, Any]) -> str:
        prompt = f"""
You are an expert in formal methods and Solidity smart contracts, specializing in BEHAVIOR SEMANTIC ALIGNMENT.

PRIMARY GOAL: Compare the Legal FSM with the Contract FSM and identify where the CONTRACT'S BEHAVIOR semantically differs from the LEGAL BEHAVIOR.

======================
LEGAL FSM (Expected Behavior)
======================
{json.dumps(legal_fsm, indent=2, ensure_ascii=False)}

======================
CONTRACT FSM (Actual Behavior)
======================
{json.dumps(contract_fsm, indent=2, ensure_ascii=False)}

======================
CRITICAL: BEHAVIOR SEMANTIC ALIGNMENT
======================
When comparing FSMs, focus on BEHAVIOR SEMANTICS, not just structure:

1. AUTOMATIC vs EXPLICIT BEHAVIORS:
   - If Legal FSM says "automatically", "shall renew unless", "must transition when", the behavior is MANDATORY
   - Check if Contract FSM requires explicit function calls by specific parties (WRONG)
   - Correct implementation: ANY party can call an enforcement function to trigger the automatic behavior

   Example:
   Legal: "The Agreement shall automatically renew unless either party gives 60 days' notice"
   ❌ WRONG: function renewTerm() external {{ termEnd += RENEWAL_TERM; }}
            (Problem: Requires explicit call, renewal is optional)
   ✅ CORRECT: function enforceTerm() external {{
                 if (block.timestamp >= termEnd) {{
                   if (!noticeGiven) {{ termEnd += RENEWAL_TERM; }}  // Automatic renewal
                   else {{ terminated = true; }}
                 }}
               }}
            (Correct: Any party can trigger, behavior is based on conditions)

2. DEFAULT BEHAVIORS:
   - What happens if NO action is taken?
   - Legal default behavior MUST match Contract default behavior

   Example:
   Legal: Default is renewal (if no notice given)
   Contract: Default is nothing happens (if no one calls function)
   → BEHAVIOR_SEMANTIC_MISMATCH

3. TIME-BASED TRIGGERS:
   - If Legal FSM has time-based transitions, check if Contract uses block.timestamp
   - Check if time conditions are enforced correctly

   Example:
   Legal: "Shares are locked for 6 months after merger"
   ❌ WRONG: require(!sharesLocked, "Shares locked");  // Missing time check
   ✅ CORRECT: require(block.timestamp >= mergerDate + 180 days, "Lock-up period not expired");

4. CONDITIONAL LOGIC COMPLETENESS:
   - If Legal FSM has "if X then Y, else Z", check if Contract has both branches
   - Check if all conditions are properly implemented

5. EXECUTABILITY OF TRANSITIONS:
     - If a transition function only checks current state but lacks legal-condition guards
         (time/role/compliance/notice), report MISSING_GUARD.
     - If Legal FSM includes key obligations (e.g., Rule 145, legend removal, registration,
         counsel opinion, reporting, notice) but Contract FSM has no corresponding signals,
         report MISSING_FUNCTIONALITY.

======================
ANALYSIS INSTRUCTIONS
======================
For each issue found, output a JSON object with the following structure:
{{
  "issue_type": "<one of: MISSING_STATE, EXTRA_STATE, MISSING_TRANSITION, INVALID_TRANSITION, MISSING_GUARD, MISSING_ACTION, MISSING_FUNCTIONALITY, INCOMPLETE_ACTORS, TIMING_CONSTRAINTS_MISSING, BEHAVIOR_SEMANTIC_MISMATCH, AUTOMATIC_BEHAVIOR_MISSING, DEFAULT_BEHAVIOR_MISMATCH>",
  "severity": "<one of: critical, high, medium, low>",
  "category": "<one of: state, transition, guard, action, functionality, actor, timing, behavior>",
  "description": "<clear description of the behavior semantic issue>",
  "location": "<specific location if identifiable, e.g., 'state X', 'transition from A to B', 'function renewTerm', or 'unknown'>",
  "suggestion": "<actionable fix suggestion focusing on behavior alignment>",
  "legal_reference": "<relevant behavior description from Legal FSM>",
  "contract_reference": "<relevant behavior description from Contract FSM or 'not found'>"
}}

SEVERITY GUIDELINES:
- critical: Core behavior missing or fundamentally misaligned, contract cannot work as legally required
- high: Important behavior misaligned, affects main business logic semantics
- medium: Secondary behavior misaligned, affects some business logic
- low: Minor behavioral details, does not affect core functionality

IMPORTANT:
1. PRIORITIZE behavior semantic issues over structural issues
2. Use BEHAVIOR_SEMANTIC_MISMATCH, AUTOMATIC_BEHAVIOR_MISSING, or DEFAULT_BEHAVIOR_MISMATCH when behavior semantics don't align
3. Provide actionable suggestions that explain HOW to align the behavior (not just "add X")
4. Only report real issues, not extraction artifacts
5. Focus on business-critical behavior alignment first
6. Return AT MOST 6 highest-confidence non-duplicate issues; omit low-confidence findings

Output ONLY a JSON array of issue objects (no explanations, no markdown):
"""
        return prompt

    # =========================
    # 解析结构化反馈（改进）
    # =========================
    def _parse_structured_feedback(self, llm_output: str) -> List[Dict[str, Any]]:
        """解析LLM输出为结构化的问题列表"""
        # 清理可能的markdown代码块
        cleaned = (llm_output or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 优先尝试直接解析完整JSON
        payload = None
        try:
            payload = json.loads(cleaned)
        except Exception:
            payload = None

        if isinstance(payload, list):
            validated = self._validate_issues_payload(payload)
            logger.info(f"Parsed {len(validated)} structured issues")
            return validated

        if isinstance(payload, dict):
            issues = payload.get("issues", [])
            if isinstance(issues, list):
                validated = self._validate_issues_payload(issues)
                logger.info(f"Parsed {len(validated)} structured issues from object payload")
                return validated

        # 次优：提取数组片段再解析
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                payload = json.loads(match.group())
                if isinstance(payload, list):
                    validated = self._validate_issues_payload(payload)
                    logger.info(f"Parsed {len(validated)} structured issues from extracted array")
                    return validated
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structured feedback as JSON array: {e}")

        # Fallback: 解析为字符串列表并结构化
        logger.warning("Falling back to simple string parsing")
        simple_issues = self._parse_feedback(llm_output)
        converted = []
        for issue in simple_issues:
            if isinstance(issue, dict):
                converted.extend(self._validate_issues_payload([issue]))
            else:
                converted.append(self._convert_string_to_structured(str(issue)))
        return converted

    def _validate_issues_payload(self, issues: List[Any]) -> List[Dict[str, Any]]:
        """标准化并校验问题结构。"""
        validated_issues = []
        for issue in issues:
            if isinstance(issue, str):
                validated_issues.append(self._convert_string_to_structured(issue))
                continue
            if not isinstance(issue, dict):
                continue

            issue_type = issue.get("issue_type", "UNKNOWN")
            severity = issue.get("severity", "medium")
            category = issue.get("category", "functionality")

            if issue_type not in ISSUE_TYPES:
                issue_type = "UNKNOWN"
            if severity not in SEVERITY_LEVELS:
                severity = "medium"
            if category not in CATEGORY_TYPES:
                category = "functionality"

            validated_issue = {
                "issue_type": issue_type,
                "severity": severity,
                "category": category,
                "description": str(issue.get("description", "")).strip(),
                "location": str(issue.get("location", "unknown")).strip() or "unknown",
                "suggestion": str(issue.get("suggestion", "")).strip(),
                "legal_reference": str(issue.get("legal_reference", "")).strip(),
                "contract_reference": str(issue.get("contract_reference", "not found")).strip() or "not found"
            }
            validated_issues.append(validated_issue)
        return validated_issues

    def _detect_rule_based_semantic_issues(self, legal_fsm: Dict[str, Any], contract_fsm: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于规则补充行为语义问题，减少纯LLM输出的不稳定性。"""
        issues: List[Dict[str, Any]] = []

        legal_text = json.dumps(legal_fsm, ensure_ascii=False).lower()
        contract_text = json.dumps(contract_fsm, ensure_ascii=False).lower()

        legal_has_automatic = any(k in legal_text for k in [
            "automatically", "shall renew unless", "must transition", "unless"
        ])
        legal_has_default = any(k in legal_text for k in [
            "default", "unless", "if no", "without notice", "no notice"
        ])
        legal_has_timing = any(k in legal_text for k in [
            "time", "deadline", "day", "days", "month", "year", "window", "before", "after"
        ])

        contract_has_time_logic = any(k in contract_text for k in [
            "block.timestamp", "timestamp", "deadline", "days", "hours", "minutes"
        ])
        contract_has_enforcement_style = any(k in contract_text for k in [
            "enforce", "check", "process", "sync", "settle", "refresh"
        ])
        contract_has_branch_logic = any(k in contract_text for k in [
            " if ", " else ", "?", "noticegiven", "default"
        ])

        # 过渡函数可执行性检测：有状态迁移但缺少非状态法律guard
        transitions = contract_fsm.get("fsm", {}).get("transitions", []) if isinstance(contract_fsm, dict) else []
        if not isinstance(transitions, list):
            transitions = []

        def _is_state_only_guard(g: str) -> bool:
            compact = re.sub(r"\s+", "", str(g or "").lower()).strip("()")
            if not compact:
                return False
            return bool(
                re.fullmatch(r"currentstate==[a-z0-9_]+\.[a-z0-9_]+", compact)
                or re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+==currentstate", compact)
            )

        guard_gap_count = 0
        for t in transitions:
            if not isinstance(t, dict):
                continue

            from_state = t.get("from")
            to_state = t.get("to")
            if not from_state or not to_state:
                continue

            fn = str(t.get("function", "unknown")).strip() or "unknown"
            guards = t.get("guards", [])
            if isinstance(guards, str):
                guards = [guards]
            if not isinstance(guards, list):
                guards = []
            guards = [str(g).strip() for g in guards if str(g).strip()]

            has_non_state_guard = any(not _is_state_only_guard(g) for g in guards)
            if not guards or not has_non_state_guard:
                guard_gap_count += 1
                if guard_gap_count <= 3:
                    issues.append({
                        "issue_type": "MISSING_GUARD",
                        "severity": "high",
                        "category": "guard",
                        "description": "Transition has state-only guards and misses legal-condition guards.",
                        "location": f"function {fn}: {from_state} -> {to_state}",
                        "suggestion": "Add at least one non-state legal guard (time/role/compliance/notice) before state transition.",
                        "legal_reference": "Legal transitions are condition-bound, not state-only.",
                        "contract_reference": f"Transition {from_state}->{to_state} in function {fn}",
                    })

        # 关键业务功能信号覆盖检测（仍由Agent4规则产生反馈）
        keyword_topics = [
            ("rule_145", ["rule 145", "145(d)", "rule145"], "Rule 145 transfer compliance workflow"),
            ("legend_removal", ["legend", "reissue", "certificate"], "legend removal/reissue workflow"),
            ("registration", ["registration statement", "registered"], "registration-based transfer path"),
            ("counsel_opinion", ["counsel", "legal opinion", "opinion"], "counsel-opinion based exemption path"),
            ("section_13_reporting", ["section 13", "report", "filing"], "post-merger reporting obligations"),
            ("notice_workflow", ["notice", "certified mail", "registered mail"], "formal notice workflow"),
        ]

        missing_topic_count = 0
        for _, kws, desc in keyword_topics:
            legal_has = any(k in legal_text for k in kws)
            contract_has = any(k in contract_text for k in kws)
            if legal_has and not contract_has:
                missing_topic_count += 1
                if missing_topic_count <= 3:
                    issues.append({
                        "issue_type": "MISSING_FUNCTIONALITY",
                        "severity": "high",
                        "category": "functionality",
                        "description": f"Legal FSM requires {desc}, but Contract FSM has no corresponding behavior signal.",
                        "location": "behavior-level",
                        "suggestion": f"Implement an explicit {desc} path with guards/actions consistent with legal constraints.",
                        "legal_reference": f"Legal constraints include keywords: {', '.join(kws)}",
                        "contract_reference": "No corresponding keyword/functionality signal found",
                    })

        if legal_has_automatic and not (contract_has_time_logic or contract_has_enforcement_style):
            issues.append({
                "issue_type": "AUTOMATIC_BEHAVIOR_MISSING",
                "severity": "critical",
                "category": "behavior",
                "description": "Legal FSM implies mandatory automatic behavior, but Contract FSM lacks time-driven/enforcement-style transition logic.",
                "location": "behavior-level",
                "suggestion": "Add an enforcement function callable by any party and gate transition by objective conditions (e.g., time + notice flags).",
                "legal_reference": "Automatic or unless-style legal behavior detected",
                "contract_reference": "No timestamp/enforcement signal found"
            })

        if legal_has_default and not contract_has_branch_logic:
            issues.append({
                "issue_type": "DEFAULT_BEHAVIOR_MISMATCH",
                "severity": "high",
                "category": "behavior",
                "description": "Legal FSM encodes a default path, but Contract FSM does not expose explicit branch logic for no-action/default scenarios.",
                "location": "behavior-level",
                "suggestion": "Add explicit default branch handling (e.g., if no notice then renew else terminate).",
                "legal_reference": "Unless/default-style legal behavior detected",
                "contract_reference": "No clear default branch signal found"
            })

        if legal_has_timing and not contract_has_time_logic:
            issues.append({
                "issue_type": "TIMING_CONSTRAINTS_MISSING",
                "severity": "high",
                "category": "timing",
                "description": "Legal FSM includes timing constraints, but Contract FSM lacks explicit time checks.",
                "location": "timing-level",
                "suggestion": "Introduce explicit time guards (e.g., block.timestamp with deadlines/windows) in related transitions.",
                "legal_reference": "Timing-related legal constraints detected",
                "contract_reference": "No timing guard signal found"
            })

        return issues

    def _merge_dedup_issues(self, llm_issues: List[Dict[str, Any]], rule_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并LLM与规则问题并去重，重复项保留更高严重级别。"""
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for issue in (llm_issues or []) + (rule_issues or []):
            key = (
                issue.get("issue_type", "UNKNOWN"),
                issue.get("location", "unknown"),
                issue.get("description", "").strip().lower()
            )
            existing = merged.get(key)
            if not existing:
                merged[key] = issue
                continue
            if SEVERITY_LEVELS.get(issue.get("severity", "low"), 0) > SEVERITY_LEVELS.get(existing.get("severity", "low"), 0):
                merged[key] = issue
        return list(merged.values())

    def _convert_string_to_structured(self, issue_str: str) -> Dict[str, Any]:
        """将简单字符串转换为结构化格式（fallback）"""
        # 尝试从字符串中提取issue_type
        issue_type = "UNKNOWN"
        for itype in ISSUE_TYPES.keys():
            if itype.replace("_", " ").lower() in issue_str.lower():
                issue_type = itype
                break

        # 默认严重程度为medium
        severity = "medium"
        if "critical" in issue_str.lower() or "missing state" in issue_str.lower():
            severity = "high"
        elif "minor" in issue_str.lower() or "extra" in issue_str.lower():
            severity = "low"

        return {
            "issue_type": issue_type,
            "severity": severity,
            "category": "functionality",
            "description": issue_str,
            "location": "unknown",
            "suggestion": "Please review and fix this issue",
            "legal_reference": "",
            "contract_reference": "not found"
        }

    # =========================
    # 智能过滤误报
    # =========================
    def _filter_false_positives(self, issues: List[Dict[str, Any]], legal_fsm: Dict[str, Any], contract_fsm: Dict[str, Any]) -> List[Dict[str, Any]]:
        """过滤可能的误报（由于Agent3提取不完整导致）"""
        filtered = []

        for issue in issues:
            # 如果是MISSING_STATE，检查是否真的缺失
            if issue["issue_type"] == "MISSING_STATE":
                # 从description中提取状态名
                state_name = self._extract_state_name(issue["description"])
                if state_name:
                    # 检查Contract FSM中是否真的没有这个状态
                    contract_states = contract_fsm.get("fsm", {}).get("states", [])
                    if any(state_name.lower() in str(s).lower() for s in contract_states):
                        logger.debug(f"Filtered false positive: {issue['description']}")
                        continue

            # 如果是MISSING_GUARD，检查是否可能是提取不完整
            if issue["issue_type"] == "MISSING_GUARD":
                # 如果Contract FSM的guards列表为空，可能是提取问题，降低严重程度
                if issue["severity"] == "critical":
                    issue["severity"] = "high"

            filtered.append(issue)

        logger.info(f"Filtered {len(issues) - len(filtered)} false positives")
        return filtered

    def _extract_state_name(self, description: str) -> Optional[str]:
        """从描述中提取状态名"""
        import re
        # 尝试匹配 'state X' 或 'STATE_X' 模式
        match = re.search(r"state['\s]+([A-Z_]+)", description, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    # =========================
    # 按严重程度排序
    # =========================
    def _sort_by_severity(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按严重程度排序问题（critical > high > medium > low）"""
        return sorted(issues, key=lambda x: SEVERITY_LEVELS.get(x["severity"], 0), reverse=True)

    # =========================
    # 生成统计信息
    # =========================
    def _generate_statistics(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成问题统计信息"""
        stats = {
            "total": len(issues),
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "by_issue_type": defaultdict(int)
        }

        for issue in issues:
            stats["by_severity"][issue["severity"]] += 1
            stats["by_category"][issue["category"]] += 1
            stats["by_issue_type"][issue["issue_type"]] += 1

        # 转换为普通dict
        stats["by_severity"] = dict(stats["by_severity"])
        stats["by_category"] = dict(stats["by_category"])
        stats["by_issue_type"] = dict(stats["by_issue_type"])

        # 确保所有严重程度都有值
        for severity in ["critical", "high", "medium", "low"]:
            if severity not in stats["by_severity"]:
                stats["by_severity"][severity] = 0

        return stats

    # =========================
    # 格式化为字符串（向后兼容）
    # =========================
    def _format_issue_as_string(self, issue: Dict[str, Any]) -> str:
        """将结构化问题格式化为字符串（向后兼容）"""
        severity_prefix = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🟢 LOW"
        }.get(issue["severity"], "")

        return f"{severity_prefix}: {issue['issue_type']}: {issue['description']}"


    # =========================
    # 解析简单反馈
    # =========================
    def _parse_feedback(self, llm_output: str) -> List[str]:
        """解析简单的字符串列表反馈（fallback方法）"""
        # 尝试提取 JSON 数组
        match = re.search(r"\[[\s\S]*\]", llm_output or "")
        if match:
            try:
                payload = json.loads(match.group())
                if isinstance(payload, list):
                    return [x for x in payload if str(x).strip()]
            except Exception:
                pass

        # fallback: 只抽取“看起来像问题描述”的行，避免把JSON字段逐行误判为问题
        candidate_lines: List[str] = []
        for raw_line in (llm_output or "").splitlines():
            line = raw_line.strip()
            if not line or len(line) < 12:
                continue

            lower = line.lower()
            if line in {"[", "]", "{", "}", "},"}:
                continue

            # 过滤 JSON 键值行（否则会造成问题数暴涨）
            if re.match(r'^"(issue_type|severity|category|description|location|suggestion|legal_reference|contract_reference)"\s*:', line):
                continue
            if re.match(r'^(issue_type|severity|category|description|location|suggestion|legal_reference|contract_reference)\s*:', lower):
                continue

            # 仅保留条目行或包含问题关键词的描述行
            is_bullet = bool(re.match(r'^(-|\*|\d+[\.)])\s+', line))
            has_issue_keyword = any(
                kw in lower
                for kw in [
                    "mismatch", "missing", "invalid", "constraint", "behavior",
                    "transition", "state", "guard", "action", "default", "automatic", "timing"
                ]
            )
            if not (is_bullet or has_issue_keyword):
                continue

            cleaned = re.sub(r'^(-|\*|\d+[\.)])\s+', '', line).strip()
            if cleaned:
                candidate_lines.append(cleaned)

        # 去重+截断，防止一次回退注入过多噪声问题
        issues: List[str] = []
        seen = set()
        for item in candidate_lines:
            key = re.sub(r"\s+", " ", item).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            issues.append(item)
            if len(issues) >= 15:
                break

        return issues

    # =========================
    # 保存反馈到本地
    # =========================
    def _save_feedback(self, contract_id: str, feedback_result: Dict[str, Any]):
        out_path = os.path.join(
            self.output_dir,
            f"{contract_id}_feedback.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(feedback_result, f, ensure_ascii=False, indent=2)
        logger.info(f"Agent4 feedback saved to {out_path}")
