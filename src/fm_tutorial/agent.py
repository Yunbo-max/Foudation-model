"""Bounded callable agent runner with auditable, externally verified traces.

This is an orchestration example, not an Agent RL trainer or a benchmark
environment. It never interprets shell commands or evaluates policy text.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any


_MAX_LOG_DEPTH = 32


def _type_summary(value: Any, reason: str) -> str:
    """Use a bounded type label, never potentially recursive user-defined repr."""
    return f"<{type(value).__name__}: {reason}>"


def _json_safe(value: Any, ancestors: frozenset[int] = frozenset(), depth: int = 0) -> Any:
    """Bound nesting before JSON/deepcopy; summarize unsupported or broken values."""
    try:
        if depth >= _MAX_LOG_DEPTH:
            return _type_summary(value, "maximum log depth reached")
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        # Preserve the established simple-byte observation display. The exact
        # built-in type has no arbitrary user-defined repr to recurse or block.
        if type(value) is bytes:
            return repr(value)
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in ancestors:
                return "<cycle>"
            ancestors = ancestors | {id(value)}
            if isinstance(value, Mapping):
                return {
                    key if isinstance(key, str) else (str(key) if key is None or type(key) in (bool, int, float, bytes) else _type_summary(key, "non-string key")):
                    _json_safe(item, ancestors, depth + 1)
                    for key, item in value.items()
                }
            return [_json_safe(item, ancestors, depth + 1) for item in value]
        return _type_summary(value, "unsupported JSON value")
    except RecursionError:
        return _type_summary(value, "serialization recursion")
    except Exception as error:
        # Custom Mapping access can fail too; do not call error/value repr or str.
        return _type_summary(value, f"serialization failed ({type(error).__name__})")


def run_agent(
    policy: Callable[[list[dict[str, Any]]], dict[str, Any]],
    tools: Mapping[str, Callable[..., Any]],
    verifier: Callable[[Any, list[dict[str, Any]]], bool],
    *,
    max_steps: int = 8,
    time_budget_seconds: float = 30.0,
    self_judge: Callable[[Any, list[dict[str, Any]]], Any] | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a bounded trajectory and return status, answer, trace and steps.

    policy(history) returns either:
      {"type": "tool", "tool": "name", "arguments": {"keyword": value}}
      {"type": "final", "answer": JSON_serializable_value}
    Registered tools receive keyword arguments. History contains JSON-safe
    observations, and each callback receives a copy so it cannot rewrite the
    audit trail. Unknown tools, malformed actions and tool errors consume a
    step and become observations for the next policy call.
    Logged nesting is limited to 32 levels; deeper or unsupported observations
    become type summaries, so JSON encoding and history copying stay bounded
    in depth. Arbitrary observation repr methods are never invoked.

    verifier(answer,history) must return a bool and is the sole authority for
    success. self_judge is advisory even when it claims success or raises.
    A final action terminates as success or verification_failed. Other statuses
    are max_steps, time_budget and policy_error. JSONL, if requested, records each
    action followed by one termination record; an existing file is overwritten.

    The deadline is cooperative: checked before/after callbacks, with no forced
    interruption of blocking Python code. Real environments must implement
    their own tool/network/process timeouts and isolation. No long-horizon
    benchmark or RL capability follows from this control-flow demonstration.
    """
    if not callable(policy) or not callable(verifier):
        raise ValueError("policy and verifier must be callable")
    if not isinstance(tools, Mapping) or any(not isinstance(name, str) or not name or not callable(tool) for name, tool in tools.items()):
        raise ValueError("tools must map nonempty names to callables")
    if self_judge is not None and not callable(self_judge):
        raise ValueError("self_judge must be callable when supplied")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(time_budget_seconds, bool) or not isinstance(time_budget_seconds, (int, float)) or not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be finite and positive")
    tools = dict(tools)
    trace: list[dict[str, Any]] = []
    answer = None
    status = "max_steps"
    started = time.monotonic()
    deadline = started + time_budget_seconds
    log_file = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_file = path.open("w", encoding="utf-8")

    def expired() -> bool:
        return time.monotonic() >= deadline

    def append(event: dict[str, Any]) -> None:
        event["elapsed_seconds"] = time.monotonic() - started
        safe_event = _json_safe(event)
        trace.append(safe_event)
        if log_file is not None:
            log_file.write(json.dumps(safe_event, ensure_ascii=False, allow_nan=False) + "\n")
            log_file.flush()

    try:
        for step in range(1, max_steps + 1):
            if expired():
                status = "time_budget"
                break
            event: dict[str, Any] = {"event": "step", "step": step}
            try:
                action = policy(deepcopy(trace))
            except Exception as error:
                event["error"] = f"policy: {type(error).__name__}: {error}"
                append(event)
                status = "policy_error"
                break
            event["action"] = _json_safe(action)
            if expired():
                event["error"] = "time budget exhausted before action execution"
                append(event)
                status = "time_budget"
                break
            try:
                if not isinstance(action, dict):
                    raise ValueError("action must be a dictionary")
                action_type = action.get("type")
                if action_type == "tool":
                    name = action.get("tool")
                    arguments = action.get("arguments")
                    if not isinstance(name, str) or not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
                        raise ValueError("tool action requires a tool name and keyword argument dictionary")
                    if name not in tools:
                        raise ValueError(f"unknown tool: {name}")
                elif action_type == "final":
                    if "answer" not in action:
                        raise ValueError("final action requires answer")
                    json.dumps(action["answer"], allow_nan=False)
                else:
                    raise ValueError("action type must be tool or final")
            except (ValueError, TypeError, OverflowError, RecursionError) as error:
                event["error"] = f"invalid_action: {error}"
                event["observation"] = {"error": event["error"]}
                append(event)
                continue
            if action_type == "tool":
                try:
                    event["observation"] = tools[name](**arguments)
                except Exception as error:
                    event["error"] = f"tool: {type(error).__name__}: {error}"
                    event["observation"] = {"error": event["error"]}
                append(event)
                if expired():
                    status = "time_budget"
                    break
                continue
            answer = action["answer"]
            try:
                verified = verifier(deepcopy(answer), deepcopy(trace + [_json_safe(event)]))
                if not isinstance(verified, bool):
                    raise ValueError("verifier must return a bool")
                event["verified"] = verified
            except Exception as error:
                event["verified"] = False
                event["verification_error"] = f"{type(error).__name__}: {error}"
            if self_judge is not None and not expired():
                try:
                    event["self_judgment"] = self_judge(deepcopy(answer), deepcopy(trace + [_json_safe(event)]))
                except Exception as error:
                    event["self_judge_error"] = f"{type(error).__name__}: {error}"
            append(event)
            status = "time_budget" if expired() else ("success" if event["verified"] else "verification_failed")
            break
        result = {"status": status, "answer": _json_safe(answer), "trace": trace, "steps": len(trace)}
        if log_file is not None:
            log_file.write(json.dumps({"event": "termination", "status": status, "answer": result["answer"], "steps": result["steps"]}, ensure_ascii=False, allow_nan=False) + "\n")
            log_file.flush()
        return result
    finally:
        if log_file is not None:
            log_file.close()
