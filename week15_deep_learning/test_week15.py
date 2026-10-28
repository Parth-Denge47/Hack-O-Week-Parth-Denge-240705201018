"""Tests for Week 15. Network-dependent parts (pretrained weights, Hugging Face, LLM APIs) are only tested
offline-safe: structure and wiring, not downloads.  Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn_regularization as cr  # noqa: E402
import dl_basics as basics  # noqa: E402
import llm_apis  # noqa: E402
import sequence_models as sm  # noqa: E402


class TestBasics(unittest.TestCase):
    def test_tensor_ops(self):
        t = basics.tensor_tour()
        self.assertEqual(t["shape"], (2, 2))
        self.assertEqual(t["matmul"].tolist(), [[3.0, 3.0], [7.0, 7.0]])
        self.assertEqual(t["broadcast"].tolist(), [[11.0, 22.0], [13.0, 24.0]])

    def test_autograd_matches_analytic_gradient(self):
        g = basics.autograd_demo()
        self.assertAlmostEqual(g["dy_dx"], g["analytic"])
        self.assertEqual(g["dz_dw"], [3.0, 4.0])
        self.assertEqual(g["dz_db"], 1.0)

    def test_training_loop_recovers_line(self):
        model, losses, _ = basics.train_linear()
        self.assertAlmostEqual(model.weight.item(), 3.0, delta=0.1)
        self.assertAlmostEqual(model.bias.item(), 1.0, delta=0.1)
        self.assertLess(losses[-1], losses[0] / 10)


class TestCNN(unittest.TestCase):
    def test_output_shape_and_pooling(self):
        m = cr.CNN(batch_norm=True, dropout=0.3)
        self.assertEqual(m(torch.zeros(5, 1, 8, 8)).shape, (5, 10))
        self.assertEqual(m.features(torch.zeros(1, 1, 8, 8)).shape, (1, 32, 2, 2))  # two 2x2 poolings

    def test_dropout_only_active_in_train_mode(self):
        m = cr.CNN(dropout=0.5)
        x = torch.rand(4, 1, 8, 8)
        m.eval()
        self.assertTrue(torch.equal(m(x), m(x)))
        m.train()
        self.assertFalse(torch.equal(m(x), m(x)))

    def test_cnn_learns_digits(self):
        train, val, test = cr.get_digit_tensors()
        torch.manual_seed(0)
        m = cr.CNN()
        cr.fit(m, train, val, epochs=25)
        self.assertGreater(cr.evaluate(m, *test)[1], 0.85)

    def test_early_stopping_halts_and_restores_best(self):
        train, val, test = cr.get_digit_tensors()
        torch.manual_seed(0)
        m = cr.CNN()
        h = cr.fit(m, train, val, epochs=200, patience=3)
        self.assertLess(h["stopped_at"], 200)
        self.assertAlmostEqual(cr.evaluate(m, *val)[0], min(h["val_loss"]), places=4)


class TestSequence(unittest.TestCase):
    def test_all_cells_run_and_learn(self):
        for kind in ("rnn", "lstm", "gru"):
            _, curve, _ = sm.train_regressor(kind, epochs=25)
            self.assertLess(curve[-1], curve[0], kind)
            self.assertLess(curve[-1], 0.05, kind)

    def test_gated_cells_remember_first_token(self):
        self.assertGreater(sm.memory_task_accuracy("lstm"), 0.9)
        self.assertGreater(sm.memory_task_accuracy("gru"), 0.9)


class TestLLMWiring(unittest.TestCase):
    def test_backend_detection_priority(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"}, clear=True):
            self.assertEqual(llm_apis.detect_backend(), "anthropic")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "y"}, clear=True):
            self.assertEqual(llm_apis.detect_backend(), "openai")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(llm_apis, "_ollama_up", return_value=False):
            self.assertIsNone(llm_apis.detect_backend())

    def test_every_backend_registered_and_tasks_defined(self):
        self.assertEqual(set(llm_apis.BACKENDS), {"anthropic", "openai", "ollama", "hf"})
        self.assertEqual(set(llm_apis.TASKS), {"text generation", "summarization", "question answering"})


if __name__ == "__main__":
    unittest.main()
