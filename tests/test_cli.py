import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rag_workbench.__main__ import main
from rag_workbench.data import sha256
from tests.test_evaluate import FakeBackend

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_run_verify_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = json.loads((ROOT / "config/experiment.json").read_text())
            config["bootstrap_samples"] = 10
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            args = [
                "--corpus",
                str(ROOT / "data/corpus.json"),
                "--queries",
                str(ROOT / "data/queries.json"),
                "--config",
                str(config_path),
                "--output",
                str(output),
            ]
            hashes = {
                "corpus": sha256(ROOT / "data/corpus.json"),
                "queries": sha256(ROOT / "data/queries.json"),
                "config": sha256(config_path),
            }
            with (
                patch("rag_workbench.__main__.LocalModels", return_value=FakeBackend()),
                patch("rag_workbench.__main__.provenance", return_value={"inputs_sha256": hashes}),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(main(["run", *args]), 0)
                self.assertEqual(
                    main(["verify", *args, "--report", str(output / "report.json")]), 0
                )
                self.assertEqual(main(["run", *args]), 2)
                self.assertEqual(main(["verify", *args]), 2)
                report = json.loads((output / "report.json").read_text())
                report["provenance"]["inputs_sha256"]["corpus"] = "changed"
                (output / "report.json").write_text(json.dumps(report))
                self.assertEqual(
                    main(["verify", *args, "--report", str(output / "report.json")]), 2
                )
            self.assertEqual(len((output / "trace.jsonl").read_text().splitlines()), 36)
            self.assertTrue((output / "errors.json").is_file())

    def test_download_is_explicit(self):
        with (
            patch("rag_workbench.__main__.download_models") as download,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(["download", "--config", str(ROOT / "config/experiment.json")]), 0
            )
            download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
