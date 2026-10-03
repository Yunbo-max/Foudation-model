"""Real callable and filesystem checks for the bounded educational runner."""
import importlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
import tempfile
import time
import unittest


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("fm_tutorial.agent"), "bounded agent implementation is missing")
        self.run_agent = importlib.import_module("fm_tutorial.agent").run_agent

    def test_tool_observation_reaches_policy_and_external_verifier(self):
        def policy(history):
            if not history:
                return {"type": "tool", "tool": "add", "arguments": {"a": 2, "b": 3}}
            return {"type": "final", "answer": history[-1]["observation"]}

        result = self.run_agent(policy, {"add": lambda a, b: a + b}, lambda answer, history: answer == 5 and history[0]["observation"] == 5)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], 5)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(result["trace"]), 2)

    def test_self_judge_cannot_override_external_verification_failure(self):
        result = self.run_agent(lambda history: {"type": "final", "answer": "wrong"}, {}, lambda answer, history: False, self_judge=lambda answer, history: {"success": True})
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["trace"][-1]["self_judgment"], {"success": True})
        self.assertFalse(result["trace"][-1]["verified"])

    def test_max_steps_bounds_repeated_tool_errors(self):
        def broken_tool():
            raise RuntimeError("tool failed")

        result = self.run_agent(lambda history: {"type": "tool", "tool": "broken", "arguments": {}}, {"broken": broken_tool}, lambda answer, history: True, max_steps=3)
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(len(result["trace"]), 3)
        self.assertTrue(all("error" in event for event in result["trace"]))

    def test_malformed_action_and_missing_tool_are_bounded_observations(self):
        actions = iter([None, {"type": "tool", "tool": "missing", "arguments": {}}, {"type": "final", "answer": "resolved"}])
        result = self.run_agent(lambda history: next(actions), {}, lambda answer, history: answer == "resolved", max_steps=3)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["steps"], 3)
        self.assertIn("error", result["trace"][0])
        self.assertIn("error", result["trace"][1])

    def test_unknown_action_and_nondict_tool_arguments_do_not_crash(self):
        for action in ({"type": "execute", "code": "bad"}, {"type": "tool", "tool": "tool", "arguments": []}):
            result = self.run_agent(lambda history: action, {"tool": lambda: None}, lambda answer, history: True, max_steps=2)
            self.assertEqual(result["status"], "max_steps")
            self.assertEqual(result["steps"], 2)

    def test_policy_exception_terminates_with_observable_error(self):
        def failing_policy(history):
            raise RuntimeError("policy failed")

        result = self.run_agent(failing_policy, {}, lambda answer, history: True)
        self.assertEqual(result["status"], "policy_error")
        self.assertEqual(result["steps"], 1)
        self.assertIn("error", result["trace"][0])

    def test_verifier_error_and_nonbool_result_cannot_claim_success(self):
        def failing_verifier(answer, history):
            raise RuntimeError("verification failed")

        for verifier in (failing_verifier, lambda answer, history: {"success": True}):
            result = self.run_agent(lambda history: {"type": "final", "answer": "candidate"}, {}, verifier)
            self.assertEqual(result["status"], "verification_failed")
            self.assertFalse(result["trace"][0]["verified"])
            self.assertIn("verification_error", result["trace"][0])

    def test_self_judge_error_is_advisory(self):
        def failing_judge(answer, history):
            raise ValueError("judge failed")

        result = self.run_agent(lambda history: {"type": "final", "answer": "yes"}, {}, lambda answer, history: True, self_judge=failing_judge)
        self.assertEqual(result["status"], "success")
        self.assertIn("self_judge_error", result["trace"][0])

    def test_advisory_judge_cannot_mutate_the_verified_answer(self):
        def mutating_judge(answer, history):
            answer["value"] = "forged"
            return {"success": True}

        result = self.run_agent(lambda history: {"type": "final", "answer": {"value": "real"}}, {}, lambda answer, history: answer["value"] == "real", self_judge=mutating_judge)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], {"value": "real"})

    def test_history_cannot_be_mutated_by_policy(self):
        def policy(history):
            if not history:
                return {"type": "tool", "tool": "value", "arguments": {}}
            history[0]["observation"] = "forged"
            return {"type": "final", "answer": "candidate"}

        result = self.run_agent(policy, {"value": lambda: "real"}, lambda answer, history: history[0]["observation"] == "real")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["trace"][0]["observation"], "real")

    def test_jsonl_contains_steps_and_termination_and_serializes_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            actions = iter([{"type": "tool", "tool": "bytes", "arguments": {}}, {"type": "final", "answer": {"ok": True}}])
            result = self.run_agent(lambda history: next(actions), {"bytes": lambda: b"output"}, lambda answer, history: True, log_path=path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["event"], "termination")
            self.assertEqual(rows[-1]["status"], "success")
            self.assertEqual(rows[0]["observation"], "b'output'")
            json.dumps(result, allow_nan=False)

    def test_deeply_nested_tool_output_is_bounded_and_logged_without_crashing(self):
        nested = "leaf"
        for _ in range(1100):
            nested = [nested]
        actions = iter([{"type": "tool", "tool": "nested", "arguments": {}}, {"type": "final", "answer": "done"}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested.jsonl"
            result = self.run_agent(lambda history: next(actions), {"nested": lambda: nested}, lambda answer, history: answer == "done", max_steps=2, log_path=path)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["steps"], 2)
            observation = result["trace"][0]["observation"]
            depth = 0
            while isinstance(observation, list):
                self.assertEqual(len(observation), 1)
                observation = observation[0]
                depth += 1
            self.assertLess(depth, 100, "logging must bound depth before JSON/deepcopy recursion limits")
            self.assertIsInstance(observation, str)
            self.assertIn("list", observation)
            self.assertIn("depth", observation)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["status"], "success")
            self.assertEqual(len(rows), 3)
            json.dumps(result, allow_nan=False)

    def test_unsupported_tool_output_uses_type_summary_without_recursive_repr(self):
        class RecursiveRepr:
            calls = 0

            def __repr__(self):
                self.calls += 1
                return repr(self)

        observation = RecursiveRepr()
        result = self.run_agent(lambda history: {"type": "tool", "tool": "object", "arguments": {}}, {"object": lambda: observation}, lambda answer, history: False, max_steps=1)
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(observation.calls, 0, "logging an unsupported value must not call arbitrary recursive repr")
        self.assertIn("RecursiveRepr", result["trace"][0]["observation"])
        json.dumps(result, allow_nan=False)

    def test_mapping_serialization_recursion_becomes_a_safe_type_summary(self):
        class BrokenMapping(Mapping):
            def __iter__(self):
                raise RecursionError("mapping iterator failed")

            def __len__(self):
                return 1

            def __getitem__(self, key):
                raise KeyError(key)

        result = self.run_agent(lambda history: {"type": "tool", "tool": "mapping", "arguments": {}}, {"mapping": BrokenMapping}, lambda answer, history: False, max_steps=1)
        self.assertEqual(result["status"], "max_steps")
        self.assertIn("BrokenMapping", result["trace"][0]["observation"])
        self.assertIn("recursion", result["trace"][0]["observation"])
        json.dumps(result, allow_nan=False)

    def test_builtin_numeric_mapping_keys_keep_distinct_observations(self):
        result = self.run_agent(lambda history: {"type": "tool", "tool": "mapping", "arguments": {}}, {"mapping": lambda: {1: "one", 2: "two"}}, lambda answer, history: False, max_steps=1)
        self.assertEqual(result["trace"][0]["observation"], {"1": "one", "2": "two"})

    def test_deadline_is_checked_after_callback_returns(self):
        def slow_tool():
            time.sleep(.015)
            return "late"

        result = self.run_agent(lambda history: {"type": "tool", "tool": "slow", "arguments": {}}, {"slow": slow_tool}, lambda answer, history: True, max_steps=5, time_budget_seconds=.005)
        self.assertEqual(result["status"], "time_budget")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["trace"][0]["observation"], "late")

    def test_invalid_budgets_and_callables_rejected_before_running(self):
        policy = lambda history: {"type": "final", "answer": "ok"}
        verifier = lambda answer, history: True
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.run_agent(policy, {}, verifier, max_steps=value)
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.run_agent(policy, {}, verifier, time_budget_seconds=value)
        for kwargs in ({"policy": None, "tools": {}, "verifier": verifier}, {"policy": policy, "tools": {"broken": None}, "verifier": verifier}, {"policy": policy, "tools": {}, "verifier": None}):
            with self.assertRaises(ValueError):
                self.run_agent(**kwargs)


if __name__ == "__main__":
    unittest.main()
