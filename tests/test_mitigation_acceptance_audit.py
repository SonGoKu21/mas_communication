import copy
import json
import unittest

from run_shopping_mitigation import audit_outcome


TASK = {"task_id": "current", "product_title": "Tea", "quantity": 2}
M4 = "M4_incorrect_collective_decision"
M5 = "M5_stale_context_acceptance"
M6 = "M6_state_inconsistency"
M14 = "M14_partial_tool_or_message_result_acceptance"


def evidence():
    state = {"task_id": "current", "product_title": "Tea", "product_id": "10",
             "sku": "TEA", "requested_quantity": 2, "observed_quantity": 2,
             "cart_verified": True}
    return {**state, "evidence": json.dumps(state)}


def verdict(decision="accept", task_id="current"):
    return {"decision": decision, "task_id": task_id, "reason": "model output"}


def receipt(index=0, purpose="reobserve_cart.product_page"):
    return {"receipt_index": index, "purpose": purpose, "request_method": "GET",
            "request_url": "http://shopping.invalid/tea.html" if purpose.endswith("product_page") else
                           "http://shopping.invalid/rest/V1/guest-carts/[REDACTED]/items",
            "status_code": 200, "response_sha256": "a" * 64,
            "response_hash_source": "content", "guest_cart_id_sha256": "b" * 64,
            "timestamp": "2026-09-10T12:00:00+00:00"}


def readback_receipts(start=0):
    return [receipt(start), receipt(start + 1, "reobserve_cart.items")]


def row(topology="sequential"):
    primary_receiver = "Supervisor" if topology == "hierarchical" else "Verifier"
    return {"topology": topology, "fault_type": "omission", "mitigation_mode": "guarded_recheck",
            "primary_receiver": primary_receiver,
            "decision_evidence_receiver": "Supervisor" if topology == "hierarchical" else "Coordinator",
            "primary_evidence": evidence(), "decision_evidence": evidence(),
            "environment_state": evidence(),
            "verification_verdict": None if topology == "hierarchical" else verdict(),
            "final_answer": {"verdict": verdict(), "cart_verified": True},
            "final_task_success": True, "mitigation_events": [], "events": [],
            "http_receipts": readback_receipts()}


def mitigation_event(receiver="Verifier"):
    return {"receiver": receiver, "before": None, "before_issues": ["non_delivery"],
            "triggered": True, "readback_called": True, "readback_count": 1,
            "readback_response": evidence(), "readback_receipt_indices": [0, 1],
            "replacement_used": True, "after": evidence(), "after_issues": []}


def common_recovery(value, *, before_verdict=None):
    start = len(value["http_receipts"])
    receipts = readback_receipts(start)
    value["http_receipts"].extend(copy.deepcopy(receipts))
    event = {"before_decision_evidence": copy.deepcopy(value["decision_evidence"]),
             "before_primary_evidence": copy.deepcopy(value["primary_evidence"]),
             "before_verification_verdict": copy.deepcopy(value["verification_verdict"]),
             "before_verdict": copy.deepcopy(before_verdict or verdict("reject")),
             "readback": evidence(), "readback_called": True,
             "readback_receipts": receipts, "readback_receipt_indices": [start, start + 1],
             "replacement_used": True,
             "after_verdict": verdict(),
             "receiver": "Supervisor" if value["topology"] == "hierarchical" else "Coordinator"}
    value["common_recovery_events"] = [event]
    value["decision_evidence"] = evidence()
    value["decision_evidence_receiver"] = "Supervisor" if value["topology"] == "hierarchical" else "Coordinator"
    value["final_answer"]["verdict"] = verdict()
    return event


class AcceptanceAuditTests(unittest.TestCase):
    def test_final_verdict_requires_current_task_binding(self):
        for bad in (None, "", "previous"):
            for decision in ("accept", "reject"):
                with self.subTest(task_id=bad, decision=decision):
                    value = row()
                    value["final_answer"]["verdict"] = verdict(decision, bad)
                    if decision == "reject":
                        value["decision_evidence"] = None
                    result = audit_outcome(value, TASK)
                    self.assertFalse(result["final_task_success"])
                    self.assertFalse(result["final_decision_correct"])
                    self.assertIn(M4, result["observed_M_consequence"])

    def test_wrong_task_verifier_is_not_validated_or_credited(self):
        for bad in (None, "", "previous"):
            with self.subTest(task_id=bad):
                value = row("flat")
                value["verification_verdict"] = verdict(task_id=bad)
                value["mitigation_events"] = [mitigation_event()]
                result = audit_outcome(value, TASK)
                self.assertTrue(result["final_task_success"])
                self.assertIn("incorrect_verification", result["semantic_consequences"])
                self.assertFalse(result["recovery_detected"])
                self.assertNotIn(M4, result["observed_M_consequence"])

    def test_raw_verdicts_and_input_are_preserved_without_aliases(self):
        value = row()
        value["final_answer"]["verdict"] = verdict(task_id="previous")
        original = copy.deepcopy(value)
        result = audit_outcome(value, TASK)
        self.assertEqual(value, original)
        self.assertEqual(result["final_answer"]["verdict"], original["final_answer"]["verdict"])
        self.assertEqual(result["verification_verdict"], original["verification_verdict"])
        result["legacy_evaluation"]["final_answer"]["verdict"]["reason"] = "changed copy"
        self.assertEqual(value, original)

    def test_sequential_forwarded_replacement_can_recover_after_verifier_rejects(self):
        for delivered in (True, False):
            with self.subTest(delivered=delivered):
                value = row()
                value["verification_verdict"] = verdict("reject")
                value["mitigation_events"] = [mitigation_event()]
                trace = {"source_agent": "Verifier", "target_agent": "Coordinator", "step_index": 5,
                         "fault_type": "none", "fault_applied": False,
                         "original_message": {"evidence": evidence(), "verdict": verdict("reject")}}
                if delivered:
                    trace["delivered_message"] = [copy.deepcopy(trace["original_message"])]
                value["events"] = [trace]
                result = audit_outcome(value, TASK)
                self.assertTrue(result.get("mitigation_recovery_detected"))
                self.assertFalse(result["common_recovery_detected"])
                self.assertTrue(result["recovery_detected"])

    def test_sequential_forwarding_requires_a_delivery_trace(self):
        value = row()
        value["verification_verdict"] = verdict("reject")
        value["mitigation_events"] = [mitigation_event()]
        self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_undelivered_step5_original_does_not_prove_forwarding(self):
        value = row()
        value["verification_verdict"] = verdict("reject")
        value["mitigation_events"] = [mitigation_event()]
        value["events"] = [{"source_agent": "Verifier", "target_agent": "Coordinator", "step_index": 5,
                            "fault_type": "none", "fault_applied": False, "delivered_message": [],
                            "original_message": {"evidence": evidence(), "verdict": verdict("reject")}}]
        self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_flat_identical_direct_evidence_does_not_credit_rejected_verifier_recheck(self):
        value = row("flat")
        value["verification_verdict"] = verdict("reject")
        value["mitigation_events"] = [mitigation_event()]
        self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_flat_direct_receiver_can_use_cached_proven_readback(self):
        value = row("flat")
        value["verification_verdict"] = verdict("reject")
        source = mitigation_event()
        direct = mitigation_event("Coordinator")
        direct.update(readback_called=False, readback_response=None, readback_receipt_indices=[],
                      cached_readback_event_index=0)
        value["mitigation_events"] = [source, direct]
        result = audit_outcome(value, TASK)
        self.assertTrue(result["recovery_detected"])
        self.assertEqual(result["recovery_evidence"]["mitigation_event_indices"], [1])

    def test_added_recovery_requires_actual_readback_proof(self):
        for change in ({"readback_called": False}, {"readback_response": None},
                       {"readback_error": "TimeoutError"}, {"replacement_used": False}):
            with self.subTest(change=change):
                value = row()
                value["mitigation_events"] = [{**mitigation_event(), **change}]
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_added_recovery_requires_valid_receipt_links(self):
        for indices in (None, [], [0], [-1, 0], [0, 99], [0, 0], [False, 1]):
            with self.subTest(indices=indices):
                value = row()
                value["mitigation_events"] = [{**mitigation_event(), "readback_receipt_indices": indices}]
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_recovery_receipts_must_be_successful_hashed_readback_gets(self):
        changes = [{"request_method": "POST"}, {"status_code": 500}, {"status_code": None},
                   {"response_sha256": None}, {"response_sha256": "not-a-hash"},
                   {"purpose": "add_to_cart.product_page"}, {"receipt_index": 99}]
        for change in changes:
            for common in (False, True):
                with self.subTest(change=change, common=common):
                    value = row()
                    if common:
                        event = common_recovery(value)
                        event["readback_receipts"][0].update(change)
                        value["http_receipts"][2].update(change)
                    else:
                        value["mitigation_events"] = [mitigation_event()]
                        value["http_receipts"][0].update(change)
                    self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_cached_readback_requires_an_explicit_prior_source_link(self):
        for link in (None, -1, 1, True):
            with self.subTest(link=link):
                value = row("flat")
                value["verification_verdict"] = verdict("reject")
                direct = {**mitigation_event("Coordinator"), "readback_called": False,
                          "readback_response": None, "readback_receipt_indices": [],
                          "cached_readback_event_index": link}
                value["mitigation_events"] = [mitigation_event(), direct]
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_common_recovery_is_separate_from_added_mitigation(self):
        for topology in ("sequential", "flat", "hierarchical"):
            with self.subTest(topology=topology):
                value = row(topology)
                value["mitigation_mode"] = "baseline"
                value["decision_evidence"] = None
                common_recovery(value)
                result = audit_outcome(value, TASK)
                self.assertTrue(result.get("common_recovery_detected"))
                self.assertFalse(result["mitigation_recovery_detected"])
                self.assertFalse(result["mitigation_detected"])
                self.assertEqual(result["recovery_type"], "common_live_readback")
                self.assertEqual(result["recovery_evidence"]["common_recovery_event_indices"], [0])
                self.assertEqual(result["propagation_class"], "detected_and_recovered")

    def test_common_recovery_keeps_pre_recovery_accepted_bad_evidence(self):
        cases = [("stale", M5), ("quantity", M6), ("partial", M14)]
        for kind, consequence in cases:
            for accepting_role in ("verifier", "final"):
                with self.subTest(kind=kind, accepting_role=accepting_role):
                    value = row()
                    bad = evidence()
                    if kind == "stale":
                        bad["task_id"] = "previous"
                    elif kind == "quantity":
                        bad["observed_quantity"] = 3
                    else:
                        bad = {"task_id": "current", "cart_verified": True}
                    value["primary_evidence"] = copy.deepcopy(bad)
                    value["decision_evidence"] = copy.deepcopy(bad)
                    value["verification_verdict"] = verdict("accept" if accepting_role == "verifier" else "reject")
                    before = verdict("reject") if accepting_role == "verifier" else verdict(task_id="previous")
                    event = common_recovery(value, before_verdict=before)
                    value["primary_evidence"] = evidence()
                    value["verification_verdict"] = verdict()
                    original = copy.deepcopy(event)
                    result = audit_outcome(value, TASK)
                    self.assertIn(consequence, result["observed_M_consequence"])
                    self.assertNotIn(M4, result["observed_M_consequence"])
                    self.assertTrue(result["common_recovery_detected"])
                    self.assertEqual(result["common_recovery_events"][0], original)

    def test_rejected_bad_evidence_is_not_retrospectively_accepted(self):
        value = row()
        value["primary_evidence"] = value["decision_evidence"] = {"task_id": "previous"}
        value["verification_verdict"] = verdict("reject")
        common_recovery(value)
        value["primary_evidence"] = evidence()
        value["verification_verdict"] = verdict()
        result = audit_outcome(value, TASK)
        self.assertNotIn(M5, result["observed_M_consequence"])
        self.assertNotIn(M14, result["observed_M_consequence"])

    def test_common_recovery_requires_trace_replacement_and_bound_acceptance(self):
        changes = [{"readback_called": False}, {"readback_receipts": []},
                   {"readback_receipts": [None]}, {"readback_receipts": [{}]}, {"replacement_used": False},
                   {"readback": None}, {"after_verdict": verdict(task_id="previous")},
                   {"after_verdict": verdict("reject")}, {"before_verdict": verdict()}]
        for change in changes:
            with self.subTest(change=change):
                value = row()
                event = common_recovery(value)
                event.update(change)
                result = audit_outcome(value, TASK)
                self.assertFalse(result["recovery_detected"])

    def test_common_recovery_requires_readback_to_reach_final_receiver(self):
        for change in ({"decision_evidence_receiver": "Verifier"}, {"decision_evidence": None}):
            with self.subTest(change=change):
                value = row()
                common_recovery(value)
                value.update(change)
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_common_recovery_requires_pre_recovery_snapshots(self):
        for key in ("before_verdict", "before_decision_evidence", "before_primary_evidence",
                    "before_verification_verdict"):
            with self.subTest(key=key):
                value = row()
                event = common_recovery(value)
                del event[key]
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_common_recovery_event_must_name_the_actual_final_receiver(self):
        for receiver in (None, "Verifier", "Supervisor"):
            with self.subTest(receiver=receiver):
                value = row()
                event = common_recovery(value)
                event["receiver"] = receiver
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_common_recovery_receipts_must_match_the_run_receipt_links(self):
        for change in ({"readback_receipt_indices": []}, {"readback_receipt_indices": [0, 1]},
                       {"readback_receipt_indices": [2, 99]}):
            with self.subTest(change=change):
                value = row()
                event = common_recovery(value)
                event.update(change)
                self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_common_recovery_cannot_reuse_added_mitigation_receipts(self):
        value = row()
        value["mitigation_events"] = [mitigation_event()]
        event = common_recovery(value)
        event["readback_receipt_indices"] = [0, 1]
        event["readback_receipts"] = copy.deepcopy(value["http_receipts"][:2])
        self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_pre_recovery_wrong_task_verifier_is_retained_as_incorrect(self):
        value = row()
        value["verification_verdict"] = verdict(task_id="previous")
        common_recovery(value)
        value["verification_verdict"] = verdict()
        result = audit_outcome(value, TASK)
        self.assertIn("incorrect_verification", result["semantic_consequences"])
        self.assertNotIn(M4, result["observed_M_consequence"])

    def test_correct_final_rejection_has_no_m4(self):
        value = row()
        value["primary_evidence"] = value["decision_evidence"] = None
        value["verification_verdict"] = verdict("reject")
        value["final_answer"]["verdict"] = verdict("reject")
        result = audit_outcome(value, TASK)
        self.assertTrue(result["final_decision_correct"])
        self.assertFalse(result["final_task_success"])
        self.assertNotIn(M4, result["observed_M_consequence"])

    def test_non_object_raw_verdict_remains_visible_and_is_not_success(self):
        for raw in (None, [], "accept"):
            with self.subTest(raw=raw):
                value = row()
                value["final_answer"]["verdict"] = raw
                result = audit_outcome(value, TASK)
                self.assertFalse(result["final_task_success"])
                self.assertIn(M4, result["observed_M_consequence"])
                self.assertEqual(result["final_answer"]["verdict"], raw)

    def test_common_recovery_does_not_credit_multiple_readbacks(self):
        value = row()
        event = common_recovery(value)
        value["common_recovery_events"].append(copy.deepcopy(event))
        self.assertFalse(audit_outcome(value, TASK)["recovery_detected"])

    def test_wrong_task_final_after_common_recovery_is_still_failure(self):
        value = row()
        event = common_recovery(value)
        value["final_answer"]["verdict"] = verdict(task_id="previous")
        event["after_verdict"] = verdict(task_id="previous")
        result = audit_outcome(value, TASK)
        self.assertFalse(result["final_task_success"])
        self.assertFalse(result["recovery_detected"])
        self.assertIn(M4, result["observed_M_consequence"])

    def test_common_recovery_does_not_relabel_failed_added_mitigation_as_recovered(self):
        value = row()
        value["mitigation_events"] = [mitigation_event()]
        common_recovery(value)
        result = audit_outcome(value, TASK)
        self.assertTrue(result.get("common_recovery_detected"))
        self.assertFalse(result["mitigation_recovery_detected"])


class WorkflowAcceptanceAuditTests(unittest.TestCase):
    def run_workflow(self, topology, mode, *, reject_first_final, bad_worker=False):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        import run_webarena_architecture_rq2 as workflow
        from mas_faults.shopping_mitigation import EvidencePolicy

        class Executor:
            def __init__(self, base_url):
                self.http_receipts = []
                self.readback_http_request_count = 0

            def add_to_cart(self, action):
                return evidence()

            def reobserve_cart(self, task):
                self.readback_http_request_count = 2
                self.http_receipts.extend(readback_receipts(len(self.http_receipts)))
                return evidence()

        class Client:
            call_count = prompt_tokens = completion_tokens = final_calls = 0
            model_info = SimpleNamespace(model="unit-test-only", provider="unit-test")

            def complete(self, prompt):
                self.call_count += 1
                if "action MUST" in prompt:
                    return json.dumps({"action": "add_to_cart"})
                if "Convert the delivered" in prompt:
                    payload = evidence()
                    if bad_worker:
                        payload["observed_quantity"] = 3
                    return json.dumps(payload)
                if "independent verifier" in prompt:
                    return json.dumps(verdict("reject"))
                self.final_calls += 1
                return json.dumps(verdict("reject" if reject_first_final and self.final_calls == 1 else "accept"))

        task = {**TASK, "product_url": "http://shopping.invalid/tea.html"}
        with patch.object(workflow, "ShoppingHTTPExecutor", Executor):
            return asyncio.run(workflow.run_one(Client(), task, topology, "omission", 4, 1,
                               "http://shopping.invalid", receiver_policy=EvidencePolicy(mode, task),
                               common_recovery=True))

    def test_actual_workflow_common_recovery_is_credited_in_all_arms(self):
        for topology in ("sequential", "flat", "hierarchical"):
            for mode in ("baseline", "always_recheck", "guarded_recheck"):
                with self.subTest(topology=topology, mode=mode):
                    raw = self.run_workflow(topology, mode, reject_first_final=True)
                    result = audit_outcome(raw, TASK)
                    self.assertTrue(result["common_recovery_detected"])
                    self.assertFalse(result["mitigation_recovery_detected"])
                    self.assertTrue(result["final_task_success"])
                    self.assertEqual(result["final_answer"]["verdict"], raw["final_answer"]["verdict"])

    def test_actual_workflow_forwarding_is_distinct_from_flat_bypass(self):
        for topology, recovered in (("sequential", True), ("flat", False), ("hierarchical", True)):
            with self.subTest(topology=topology):
                raw = self.run_workflow(topology, "guarded_recheck", reject_first_final=False)
                result = audit_outcome(raw, TASK)
                self.assertEqual(result["mitigation_recovery_detected"], recovered)
                self.assertFalse(result["common_recovery_detected"])

    def test_actual_workflow_cached_readback_uses_the_recorded_source_link(self):
        raw = self.run_workflow("flat", "guarded_recheck", reject_first_final=False, bad_worker=True)
        result = audit_outcome(raw, TASK)
        self.assertTrue(result["mitigation_recovery_detected"])
        self.assertEqual(result["recovery_evidence"]["mitigation_event_indices"], [1])


if __name__ == "__main__":
    unittest.main()
