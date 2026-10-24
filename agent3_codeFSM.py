import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from llm_router import call_model  # 与 Agent1/2统一接口

logger = logging.getLogger("Agent3")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - Agent3 - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

class Agent3ContractToFSM:
    def __init__(self, model_name: Optional[str]=None, output_dir: str="./Outputs/agent3_fsm"):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, contract_id: str, solidity_code: str) -> Dict[str, Any]:
        logger.info(f"Parsing Solidity contract for contract_id={contract_id}")
        states, initial_state, state_var = self._extract_states(solidity_code)
        transitions = self._extract_transitions(solidity_code, state_var)
        if self.model_name:
            transitions = self._llm_assisted_parse(solidity_code, transitions)
        fsm_result = {
            "contract_id": contract_id,
            "fsm": {
                "states": states,
                "initial_state": initial_state,
                "state_variable": state_var,
                "transitions": transitions
            },
            "extraction_confidence": {
                "states": "high" if states else "low",
                "transitions": "medium" if transitions else "low"
            },
            "warnings": []
        }
        self._save_output(contract_id, fsm_result)
        return fsm_result

    def _extract_states(self, code: str):
        enum_match = re.search(r"enum\s+(\w+)\s*\{([^\}]+)\}", code)
        states = []
        state_var = "currentState"
        initial_state = None
        if enum_match:
            state_var_enum_name = enum_match.group(1)
            states = [s.strip() for s in enum_match.group(2).split(",") if s.strip()]
            state_var_match = re.search(r"(\w+)\s+public\s+" + state_var_enum_name, code)
            if state_var_match:
                state_var = state_var_match.group(1)
            init_match = re.search(rf"{state_var}\s*=\s*{state_var_enum_name}\.(\w+)", code)
            if init_match:
                initial_state = init_match.group(1)
        return states, initial_state, state_var

    def _extract_transitions(self, code: str, state_var: str):
        """改进：提取更多上下文信息，包括函数参数、修饰符等"""
        transitions = []
        # 改进的正则表达式，捕获更多信息
        func_pattern = r"function\s+(\w+)\s*\(([^\)]*)\)\s*((?:public|external|internal|private)\s*(?:\w+\s*)*)\s*{(.*?)}"
        func_matches = re.findall(func_pattern, code, flags=re.DOTALL | re.IGNORECASE)

        for func_name, params, modifiers, body in func_matches:
            # 提取状态转换
            from_states = re.findall(rf"require\s*\(\s*{state_var}\s*==\s*\w+\.(\w+)", body)
            to_states = re.findall(rf"{state_var}\s*=\s*\w+\.(\w+)", body)

            # 改进：提取所有require语句（不仅仅是状态检查）
            guards = []
            for guard_match in re.finditer(r"require\s*\((.*?)\)", body, re.DOTALL):
                guard_text = guard_match.group(1).strip()
                # 清理多余的空白和换行
                guard_text = re.sub(r'\s+', ' ', guard_text)
                guards.append(guard_text)

            # 改进：提取所有事件和状态变更
            actions = []
            # 提取emit语句
            for emit_match in re.finditer(r"emit\s+(\w+)\s*\((.*?)\)\s*;", body, re.DOTALL):
                event_name = emit_match.group(1)
                event_params = emit_match.group(2).strip()
                actions.append(f"emit {event_name}({event_params})")

            # 提取其他重要的状态变更（如mapping更新）
            for assignment in re.finditer(r"(\w+)\[([^\]]+)\]\s*=\s*([^;]+);", body):
                var_name = assignment.group(1)
                key = assignment.group(2).strip()
                value = assignment.group(3).strip()
                actions.append(f"{var_name}[{key}] = {value}")

            from_state = from_states[0] if from_states else None
            to_state = to_states[0] if to_states else None

            transitions.append({
                "from": from_state,
                "to": to_state,
                "function": func_name,
                "parameters": params.strip(),
                "modifiers": modifiers.strip(),
                "guards": guards,
                "actions": actions
            })
        return transitions

    def _llm_assisted_parse(self, code: str, transitions: List[Dict[str, Any]]):
        prompt = f"""
You are a senior blockchain engineer analyzing Solidity smart contracts.

TASK: Enhance the extracted FSM transitions with deeper semantic analysis.

For each transition, identify:
1. All guard conditions (explicit require statements AND implicit checks like modifiers, if statements)
2. All actions and side effects (events, state changes, external calls)
3. Business logic constraints (timing requirements, access control, value checks)
4. Any missing transitions that exist in the code but weren't captured by regex

IMPORTANT:
- Do NOT invent new states that don't exist in the code
- Do NOT modify the from/to states
- ONLY enhance guards and actions with additional context

CODE:
```solidity
{code}
```

EXTRACTED TRANSITIONS (from static analysis):
{json.dumps(transitions, indent=2)}

Return a JSON array of enhanced transitions with the same structure but more complete guards and actions.
Include a "business_logic" field for each transition describing the business constraint it enforces.

Output JSON only (no explanations):
"""

        try:
            response = call_model(
                model_name=self.model_name,
                prompt=prompt,
                max_tokens=3000,
                temperature=0.1,
                expect_json=True
            )

            # 尝试解析JSON
            if isinstance(response, str):
                # 清理可能的markdown代码块
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()

                enhanced = json.loads(response)
                logger.info(f"LLM enhanced {len(enhanced)} transitions with deeper analysis")
                return enhanced
            else:
                return transitions

        except json.JSONDecodeError as e:
            logger.warning(f"LLM output could not be parsed as JSON: {e}, using static analysis only")
            return transitions
        except Exception as e:
            logger.warning(f"LLM-assisted parsing failed: {e}, using static analysis only")
            return transitions

    def _save_output(self, contract_id: str, result: Dict[str, Any]):
        out_path = os.path.join(self.output_dir, f"{contract_id}_fsm.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Agent3 FSM output saved to {out_path}")
