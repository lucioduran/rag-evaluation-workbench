import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_workbench.config import load_config
from rag_workbench.local_models import LocalModels, download_models

ROOT = Path(__file__).resolve().parents[1]


class LocalModelContractTests(unittest.TestCase):
    def test_only_pinned_safe_files_are_downloaded(self):
        config = load_config(ROOT / "config/experiment.json")
        hub = MagicMock()
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            download_models(config, Path("cache"))
        self.assertEqual(hub.snapshot_download.call_count, 2)
        for call in hub.snapshot_download.call_args_list:
            self.assertFalse(call.kwargs["token"])
            self.assertEqual(len(call.kwargs["revision"]), 40)
            self.assertNotIn("pytorch_model.bin", call.kwargs["allow_patterns"])

    def test_offline_loading_and_generation_budgets(self):
        config = load_config(ROOT / "config/experiment.json")
        torch, transformers = MagicMock(), MagicMock()
        embed_tokenizer, answer_tokenizer = MagicMock(), MagicMock()
        transformers.AutoTokenizer.from_pretrained.side_effect = [embed_tokenizer, answer_tokenizer]
        embed_tokenizer.return_value = {
            "input_ids": MagicMock(shape=(1, 4)),
            "attention_mask": MagicMock(),
        }
        answer_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 5))}
        answer_tokenizer.encode.return_value = [1, 2, 3]
        answer_tokenizer.decode.return_value = "answer"
        answer_tokenizer.eos_token_id = 1
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value.eval.return_value
        model.generate.return_value.__getitem__.return_value.tolist.return_value = [0, 7, 1]
        torch.nn.functional.normalize.return_value.tolist.return_value = [[0.6, 0.8]]
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            backend = LocalModels(config, Path("cache"))
            self.assertEqual(backend.embed([]), [])
            self.assertEqual(backend.embed(["text"]), [[0.6, 0.8]])
            self.assertEqual(backend.count_tokens("text"), 3)
            self.assertEqual(
                backend.generate("question"),
                {
                    "answer": "answer",
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "reached_token_limit": False,
                },
            )
            self.assertFalse(model.generate.call_args.kwargs["do_sample"])
            self.assertEqual(model.generate.call_args.kwargs["num_beams"], 1)
            embed_tokenizer.return_value["input_ids"].shape = (1, 999)
            with self.assertRaises(ValueError):
                backend.embed(["too long"])
            answer_tokenizer.return_value["input_ids"].shape = (1, 999)
            with self.assertRaises(ValueError):
                backend.generate("too long")
        for loader in (transformers.AutoModel, transformers.AutoModelForSeq2SeqLM):
            kwargs = loader.from_pretrained.call_args.kwargs
            self.assertTrue(kwargs["local_files_only"])
            self.assertTrue(kwargs["use_safetensors"])
            self.assertFalse(kwargs["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
