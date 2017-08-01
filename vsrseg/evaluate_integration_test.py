import unittest
import tempfile
import shutil
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import model as md
import vsrseg.load_data as ld
import evaluate as ev

class TrainerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.model = md.TestCtxBB()
        self.dataloader = DataLoader(
                TensorDataset(torch.ones((20, 4)), torch.ones((20, 8)) * 5))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_do_eval(self):
        self.model = md.TestCtxBB()
        evaluator = ev.Evaluator(cuda=[])
        ev.do_eval(evaluator, self.model, "vcoco_val", self.test_dir)
