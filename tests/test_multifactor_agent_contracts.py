import unittest

from crypto_quant.research.strategy_research.multifactor_agent_contracts import (
    AgentSessionContract,
    DESIGN_OUTPUT_SCHEMA,
    REVIEW_OUTPUT_SCHEMA,
    check_design,
    check_review,
)


def _session(**changes):
    value = {
        "schema_version": 1,
        "run_id": "agent-cycle-001",
        "experiment_budget": 3,
        "context_bytes": 32000,
    }
    value.update(changes)
    return value


def _review(**changes):
    value = {
        "summary": "The current evidence supports one focused comparison.",
        "findings": [{"claim": "The equal-weight baseline is recorded.", "evidence_refs": ["baseline-result"]}],
        "limitations": ["Historical data processing provenance is incomplete."],
    }
    value.update(changes)
    return value


def _design(**changes):
    value = {
        "action": "experiment",
        "hypothesis": "Removing one redundant card will preserve performance with lower turnover.",
        "strategy_card_ids": ["card-c", "card-a"],
        "evidence_refs": ["baseline-result"],
    }
    value.update(changes)
    return value


class AgentSessionContractTests(unittest.TestCase):
    def test_contract_is_exact_and_call_limit_is_derived(self):
        contract = AgentSessionContract.from_dict(_session())
        self.assertEqual(contract.max_model_calls, 7)
        self.assertEqual(contract.as_dict(), _session())
        self.assertEqual(set(contract.as_dict()), {
            "schema_version", "run_id", "experiment_budget", "context_bytes",
        })

        for change in (
            {"experiment_budget": 0},
            {"experiment_budget": True},
            {"experiment_budget": 1.5},
            {"context_bytes": 0},
            {"context_bytes": 32000.0},
            {"schema_version": 2},
            {"run_id": "bad run id"},
            {"max_model_calls": 8},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                AgentSessionContract.from_dict(_session(**change))

        incomplete = _session()
        del incomplete["context_bytes"]
        with self.assertRaises(ValueError):
            AgentSessionContract.from_dict(incomplete)


class AgentReplyContractTests(unittest.TestCase):
    def test_output_schemas_require_exact_structured_fields(self):
        self.assertEqual(REVIEW_OUTPUT_SCHEMA["additionalProperties"], False)
        self.assertEqual(set(REVIEW_OUTPUT_SCHEMA["required"]), {"summary", "findings", "limitations"})
        self.assertEqual(DESIGN_OUTPUT_SCHEMA["additionalProperties"], False)
        self.assertEqual(set(DESIGN_OUTPUT_SCHEMA["required"]), {
            "action", "hypothesis", "strategy_card_ids", "evidence_refs",
        })

    def test_review_requires_known_refs_for_each_finding(self):
        parsed = check_review(_review(), ["baseline-result", "input-snapshot"])
        self.assertEqual(parsed["findings"][0]["evidence_refs"], ["baseline-result"])
        self.assertEqual(check_review(_review(findings=[]), ["baseline-result"])["findings"], [])
        self.assertEqual(check_review(_review(findings=[], limitations=[]), ["baseline-result"])["limitations"], [])

        for reply in (
            _review(extra="not allowed"),
            _review(findings=[{"claim": "Unsupported reference", "evidence_refs": ["invented-record"]}]),
            _review(findings=[{"claim": "No citation", "evidence_refs": []}]),
            _review(findings=[{"claim": "Repeated citation", "evidence_refs": ["baseline-result", "baseline-result"]}]),
            _review(findings=[{"claim": "Extra nested field", "evidence_refs": ["baseline-result"], "score": 1}]),
            _review(summary="  "),
        ):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                check_review(reply, ["baseline-result"])

    def test_experiment_subset_is_known_new_and_canonical(self):
        checked = check_design(_design(), ["baseline-result"],
                               ["card-a", "card-b", "card-c", "card-d"],
                               seen_subsets={("card-b", "card-d")})
        self.assertEqual(checked["strategy_card_ids"], ["card-a", "card-c"])
        self.assertEqual(checked["evidence_refs"], ["baseline-result"])

        for reply, seen in (
            (_design(strategy_card_ids=["card-a"]), set()),
            (_design(strategy_card_ids=["card-a", "card-b", "card-c", "card-d"]), set()),
            (_design(strategy_card_ids=["card-a", "card-a"]), set()),
            (_design(strategy_card_ids=("card-a", "card-c")), set()),
            (_design(strategy_card_ids=["card-a", "other-card"]), set()),
            (_design(strategy_card_ids=["card-a", "card-c"]), {("card-c", "card-a")}),
            (_design(evidence_refs=["missing-result"]), set()),
            (_design(strategy_card_ids=["card-a", "card-b"], extra="not allowed"), set()),
            (_design(action="stop", strategy_card_ids=["card-a", "card-c"]), set()),
            (_design(action="stop", strategy_card_ids=[], evidence_refs=["missing-result"]), set()),
        ):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                check_design(reply, ["baseline-result"],
                             ["card-a", "card-b", "card-c", "card-d"], seen)

    def test_stop_requires_empty_subset_and_keeps_a_cited_reason(self):
        parsed = check_design(
            _design(action="stop", strategy_card_ids=[], hypothesis="Stop because the remaining budget is not useful."),
            ["baseline-result"], ["card-a", "card-b", "card-c"], {("card-a", "card-b")},
        )
        self.assertEqual(parsed["action"], "stop")
        self.assertEqual(parsed["strategy_card_ids"], [])

        with self.assertRaises(ValueError):
            check_design(_design(hypothesis=""), ["baseline-result"],
                         ["card-a", "card-b", "card-c"], set())


if __name__ == "__main__":
    unittest.main()
