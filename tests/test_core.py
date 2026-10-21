import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from rag_workbench.data import ContractError, Document, load_dataset
from rag_workbench.metrics import (
    answer_metrics,
    interval,
    paired_interval,
    percentile,
    retrieval_metrics,
)
from rag_workbench.retrieval import BM25, cosine_rank, fuse, rank, tokenize

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_dataset_counts_and_isolation(self):
        docs, queries = load_dataset(ROOT / "data/corpus.json", ROOT / "data/queries.json")
        self.assertEqual(len(docs), 24)
        self.assertEqual(sum(q.split == "dev" for q in queries), 12)
        self.assertEqual(sum(q.split == "eval" for q in queries), 26)
        dev = {d for q in queries if q.split == "dev" for d in q.qrels}
        evaluation = {d for q in queries if q.split == "eval" for d in q.qrels}
        self.assertFalse(dev & evaluation)

    def test_rejects_corruption_and_leakage(self):
        corpus = json.loads((ROOT / "data/corpus.json").read_text())
        labels = json.loads((ROOT / "data/queries.json").read_text())

        def unknown_doc(c, q):
            q["queries"][0]["relevance"] = {"missing": 1}

        def duplicate_doc(c, q):
            c["documents"].append(c["documents"][0])

        def overlap(c, q):
            q["queries"][0]["split"] = "eval"

        def duplicate_query(c, q):
            q["queries"][1]["id"] = q["queries"][0]["id"]

        def bad_grade(c, q):
            q["queries"][0]["relevance"] = {"archive": True}

        def bad_version(c, q):
            c["schema_version"] = True

        def extra(c, q):
            c["unexpected"] = 1

        def bad_alias(c, q):
            q["queries"][-1]["answers"] = ["42"]

        for mutate in (
            unknown_doc,
            duplicate_doc,
            overlap,
            duplicate_query,
            bad_grade,
            bad_version,
            extra,
            bad_alias,
        ):
            with self.subTest(mutation=mutate.__name__), tempfile.TemporaryDirectory() as directory:
                c, q = copy.deepcopy(corpus), copy.deepcopy(labels)
                mutate(c, q)
                root = Path(directory)
                (root / "c.json").write_text(json.dumps(c))
                (root / "q.json").write_text(json.dumps(q))
                with self.assertRaises(ContractError):
                    load_dataset(root / "c.json", root / "q.json")


class RetrievalTests(unittest.TestCase):
    def test_bm25_matches_hand_calculation(self):
        docs = (Document("a", "red", "red"), Document("b", "blue", "blue"))
        scores = dict(BM25(docs).search("red"))
        self.assertAlmostEqual(scores["a"], math.log(2) * 2 * 2.2 / 3.2)
        self.assertEqual(scores["b"], 0)
        self.assertEqual(BM25(docs).search("absent"), [("a", 0), ("b", 0)])
        self.assertEqual(tokenize("A, B42!"), ["a", "b42"])

    def test_cosine_known_vectors(self):
        result = cosine_rank(["b", "a"], [[0, 1], [1, 0]], [1, 0])
        self.assertEqual(result, [("a", 1), ("b", 0)])
        self.assertEqual(cosine_rank(["x"], [[2, 2]], [1, 1])[0][1], 0.9999999999999998)

    def test_fusion_is_rank_based_and_stable(self):
        result = fuse([["a", "b"], ["b", "a"]], 60)
        self.assertEqual([x[0] for x in result], ["a", "b"])
        self.assertAlmostEqual(result[0][1], 1 / 61 + 1 / 62)

    def test_invalid_retrieval_inputs(self):
        for call in (
            lambda: BM25(()),
            lambda: BM25((Document("x", "", ""),)),
            lambda: cosine_rank(["a"], [[0, 0]], [1, 1]),
            lambda: cosine_rank(["a"], [[1]], [1, 2]),
            lambda: cosine_rank(["a"], [[1]], [float("nan")]),
            lambda: fuse([["a", "a"]]),
            lambda: rank({"a": float("inf")}),
        ):
            with self.assertRaises(ValueError):
                call()


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics_against_hand_values(self):
        scores = retrieval_metrics(["b", "a"], {"a": 2, "c": 1}, 2)
        self.assertEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["ndcg"], (3 / math.log2(3)) / (3 + 1 / math.log2(3)))
        self.assertEqual(retrieval_metrics(["a"], {}, 3), {"recall": None, "ndcg": None})
        with self.assertRaises(ValueError):
            retrieval_metrics(["a", "a"], {"a": 1}, 2)

    def test_answer_aliases_and_multiset_f1(self):
        self.assertEqual(
            answer_metrics("The EAST kiosk.", ("east kiosk",)), {"exact_match": 1, "token_f1": 1}
        )
        self.assertEqual(answer_metrics("nine", ("9", "nine"))["exact_match"], 1)
        self.assertAlmostEqual(answer_metrics("red red blue", ("red blue",))["token_f1"], 0.8)
        self.assertEqual(answer_metrics("", ("unknown",))["token_f1"], 0)

    def test_grouped_bootstrap_reproducibility_and_pairing(self):
        result = interval([0, 0, 1, 1], ["a", "a", "b", "b"], 100, 4)
        self.assertEqual(result, interval([0, 0, 1, 1], ["a", "a", "b", "b"], 100, 4))
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(paired_interval([0, 1], [0, 1], ["a", "b"])["ci95"], [0, 0])
        self.assertEqual(percentile([0, 10], 0.5), 5)
        for call in (
            lambda: interval([], []),
            lambda: paired_interval([1], [], ["a"]),
            lambda: percentile([], 0.5),
        ):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
