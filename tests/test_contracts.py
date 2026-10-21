import copy
import json
import tempfile
import unittest
from pathlib import Path

from rag_workbench.config import load_config
from rag_workbench.data import ContractError, Document, identifier, load_dataset, read_json, text
from rag_workbench.metrics import answer_metrics, interval, percentile, retrieval_metrics
from rag_workbench.retrieval import BM25, cosine_rank, fuse

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_config_rejects_unpinned_and_invalid_settings(self):
        source = json.loads((ROOT / "config/experiment.json").read_text())
        mutations = [
            ("schema_version", True),
            ("k", 0),
            ("seed", -1),
            ("bm25_k1", 0),
            ("bm25_b", 2),
            ("bm25_b", float("nan")),
            ("prompt", "{wrong}"),
            ("prompt", None),
            ("prompt", "{question} {context!z}"),
            ("answer_model", {"id": "bad", "revision": "main"}),
            ("embedding_model", {"id": "valid/model", "revision": "main"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in mutations:
                config = copy.deepcopy(source)
                config[key] = value
                path.write_text(json.dumps(config))
                with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                    load_config(path)

    def test_text_and_json_fail_closed(self):
        for value in ("", None, "x\x00", "x\ud800", "x" + chr(0x2014)):
            with self.assertRaises(ContractError):
                text(value, "text")
        with self.assertRaises(ContractError):
            identifier("../x")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            with self.assertRaises(ContractError):
                read_json(path)
            for content in ("{", '{"x":1,"x":2}'):
                path.write_text(content)
                with self.assertRaises(ContractError):
                    read_json(path)

    def test_additional_dataset_failures(self):
        corpus = json.loads((ROOT / "data/corpus.json").read_text())
        queries = json.loads((ROOT / "data/queries.json").read_text())
        cases = [
            ("corpus", "license", "unknown"),
            ("corpus", "documents", []),
            ("query", "split", "train"),
            ("query", "answers", []),
            ("query", "question", queries["queries"][1]["question"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, key, value in cases:
                c, q = copy.deepcopy(corpus), copy.deepcopy(queries)
                (c if target == "corpus" else q["queries"][0])[key] = value
                (root / "c").write_text(json.dumps(c))
                (root / "q").write_text(json.dumps(q))
                with self.subTest(key=key), self.assertRaises(ContractError):
                    load_dataset(root / "c", root / "q")
            queries["queries"] = [q for q in queries["queries"] if q["split"] == "dev"]
            (root / "c").write_text(json.dumps(corpus))
            (root / "q").write_text(json.dumps(queries))
            with self.assertRaises(ContractError):
                load_dataset(root / "c", root / "q")

    def test_metric_and_retrieval_edge_cases(self):
        calls = [
            lambda: retrieval_metrics(["a"], {"a": True}, 1),
            lambda: answer_metrics("x", ()),
            lambda: interval([float("nan")], ["a"]),
            lambda: percentile([float("inf")], 0.5),
            lambda: BM25((Document("a", "a", "a"), Document("a", "a", "a"))),
            lambda: cosine_rank([], [], [1]),
            lambda: cosine_rank(["a"], [[1]], [0]),
            lambda: fuse([], 60),
        ]
        for call in calls:
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(answer_metrics("", ("",))["token_f1"], 1)


if __name__ == "__main__":
    unittest.main()
