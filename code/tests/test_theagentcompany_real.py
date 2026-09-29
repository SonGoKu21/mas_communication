import unittest

from mas_faults.theagentcompany_real import (
    TaskEvidence,
    TheAgentCompanyTask,
    build_evaluation_command,
    build_initialization_command,
    evaluate_task_evidence,
    intercept_task_evidence,
    task_image,
    build_agent_prompt,
    evaluate_task_evidence,
)


class TheAgentCompanyContractTests(unittest.TestCase):
    def test_clean_model_failure_is_not_classified_as_a_clean_success(self) -> None:
        evidence = TaskEvidence("finance-budget-variance", False, {"final_score": {"result": 1, "total": 4}}, "partial score")

        result = evaluate_task_evidence("clean", evidence, {"decision": "reject"}, evidence)

        self.assertEqual(result["propagation_class"], "clean_task_failure")

    def test_omission_creates_a_real_message_gap_and_m3_failure(self) -> None:
        evidence = TaskEvidence("sde-install-openjdk", True, {"final_score": {"result": 2}}, "official score 2")

        intercepted = intercept_task_evidence(evidence, "a5_omission")
        result = evaluate_task_evidence("a5_omission", intercepted.delivered, {"decision": "reject"}, evidence)

        self.assertIsNone(intercepted.delivered)
        self.assertIn("M3_incomplete_information_aggregation", result["observed_M_consequence"])
        self.assertEqual(result["propagation_class"], "propagated_to_M_final_failure")

    def test_truncation_only_assigns_m14_when_the_verifier_accepts_partial_evidence(self) -> None:
        evidence = TaskEvidence("sde-install-openjdk", True, {"final_score": {"result": 2}}, "official score 2")
        intercepted = intercept_task_evidence(evidence, "a8_truncation")

        rejected = evaluate_task_evidence("a8_truncation", intercepted.delivered, {"decision": "reject"}, evidence)
        accepted = evaluate_task_evidence("a8_truncation", intercepted.delivered, {"decision": "accept"}, evidence)

        self.assertNotIn("M14_partial_tool_or_message_result_acceptance", rejected["observed_M_consequence"])
        self.assertIn("M14_partial_tool_or_message_result_acceptance", accepted["observed_M_consequence"])

    def test_task_image_uses_official_release_name(self) -> None:
        task = TheAgentCompanyTask("sde-install-go", ())
        self.assertEqual(task_image(task), "ghcr.io/theagentcompany/sde-install-go-image:1.0.0")

    def test_initialization_uses_local_qwen_environment(self) -> None:
        task = TheAgentCompanyTask("sde-install-go", ())
        command = build_initialization_command(task, "tac-clean", "http://127.0.0.1:8004", "Qwen/Qwen3-8B")
        self.assertIn("/utils/init.sh", command)
        self.assertIn("LITELLM_BASE_URL=http://127.0.0.1:8004", command)

    def test_evaluation_uses_official_entrypoint_and_trace_path(self) -> None:
        command = build_evaluation_command("tac-clean", "/tmp/trajectory.jsonl", "/tmp/evaluation.json")
        self.assertIn("/utils/eval.py", command)
        self.assertIn("DECRYPTION_KEY=theagentcompany is all you need", command)
        self.assertIn("/tmp/trajectory.jsonl", command)

    def test_agent_prompt_requires_a_single_executable_command(self) -> None:
        prompt = build_agent_prompt("Install go 1.17")
        self.assertIn("Install go 1.17", prompt)
        self.assertIn("single shell command", prompt)


if __name__ == "__main__":
    unittest.main()
