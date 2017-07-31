import unittest
import tempfile
import shutil
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import model

class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.model = torch.nn.Linear(4, 8)
        self.dataloader = DataLoader(
                TensorDataset(torch.ones((20, 4)), torch.ones((20, 8)) * 5))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_save(self):
        self.trainer = model.BasicTrainer(
                self.model, self.dataloader, cuda=None, 
                save_dir=os.path.join(self.test_dir, "test_save"))
        # Make sure training epoch is maintained as part of state.
        self.trainer.train(2)
        outname = os.path.join(self.test_dir, "resume_train.trn")
        torch.save(self.trainer, outname)
        loaded = torch.load(outname)
        self.assertEqual(loaded.epoch, 2)

    def test_resume_train(self):
        self.trainer = model.BasicTrainer(
                self.model, self.dataloader, cuda=None, 
                save_dir=os.path.join(self.test_dir, "test_resume"))
        # Make sure epochs are consistent when training pauses.
        self.trainer.train(2)
        self.assertEqual(self.trainer.epoch, 2)
        self.trainer.train(6)
        self.assertEqual(self.trainer.epoch, 8)
        self.trainer.train(0)
        self.assertEqual(self.trainer.epoch, 8)
        self.trainer.train(3)
        self.assertEqual(self.trainer.epoch, 11)
