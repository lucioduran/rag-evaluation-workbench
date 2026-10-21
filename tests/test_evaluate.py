import copy
import unittest
from pathlib import Path

from rag_workbench.config import load_config
from rag_workbench.data import ContractError, load_dataset
from rag_workbench.evaluate import make_prompt, run, summarize
from rag_workbench.report import error_analysis, verify_report

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def embed(self, texts):
        return [[1.0, float(len(text) % 7 + 1)] for text in texts]

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        return {
            "answer": "unknown",
            "input_tokens": self.count_tokens(prompt),
            "output_tokens": 1,
            "reached_token_limit": False,
        }


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.queries = load_dataset(ROOT / "data/corpus.json", ROOT / "data/queries.json")
        cls.config = load_config(ROOT / "config/experiment.json")
        cls.config["bootstrap_samples"] = 30
        selected = tuple(q for q in cls.queries if q.split == "dev")
        cls.report = run(cls.docs, selected, FakeBackend(), cls.config)
        cls.report.update({"split": "dev", "config": cls.config})

    def test_records_cover_methods_and_rotate_order(self):
        rows = self.report["records"]
        self.assertEqual(len(rows), 36)
        self.assertEqual(
            {row["execution_position"] for row in rows if row["method"] == "bm25"}, {0, 1, 2}
        )
        self.assertTrue(all(row["input_tokens"] <= 384 for row in rows))
        verify_report(self.report, self.docs, self.queries, self.config)

    def test_prompt_budget_never_cuts_a_document(self):
        config = dict(self.config, max_input_tokens=35)
        prompt, included = make_prompt(
            self.queries[0], ["archive"], {d.id: d for d in self.docs}, FakeBackend(), config
        )
        self.assertEqual(included, [])
        self.assertNotIn("Recovery drawer", prompt)
        with self.assertRaises(ValueError):
            make_prompt(self.queries[0], [], {}, FakeBackend(), dict(config, max_input_tokens=1))

    def test_report_rejects_tampered_evidence(self):
        for field, value in [
            ("answer", "unexpected"),
            ("question", "wrong"),
            ("input_tokens", 999),
            ("total_ms", -1),
            ("context_ids", ["not-a-document"]),
        ]:
            report = copy.deepcopy(self.report)
            report["records"][0][field] = value
            if field == "answer":
                report["records"][0]["answer"] = report["records"][0]["references"][0]
            with self.subTest(field=field), self.assertRaises((ContractError, ValueError)):
                verify_report(report, self.docs, self.queries, self.config)
        report = copy.deepcopy(self.report)
        report["records"][0]["ranking"][0]["score"] += 1
        with self.assertRaises(ContractError):
            verify_report(report, self.docs, self.queries, self.config)
        report = copy.deepcopy(self.report)
        report["records"].pop()
        with self.assertRaises(ContractError):
            verify_report(report, self.docs, self.queries, self.config)

    def test_error_categories_and_alignment(self):
        errors = error_analysis(self.report["records"])
        self.assertEqual(len(errors), 36)
        self.assertTrue(all("context" in error["category"] for error in errors))
        with self.assertRaises(ValueError):
            summarize([], self.config)
        with self.assertRaises(ValueError):
            run(self.docs, (), FakeBackend(), self.config)


if __name__ == "__main__":
    unittest.main()
