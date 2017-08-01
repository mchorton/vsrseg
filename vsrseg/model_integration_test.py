import unittest
import tempfile
import shutil
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import model
import vsrseg.load_data as ld
import evaluate as ev

class TrainerIntegrationTest(unittest.TestCase):
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

    def test_pickle(self):
        self.dataloader = ld.get_loader("vcoco_train", ld.COCO_IMGDIR)
        self.trainer = model.BasicTrainer(
                self.model, self.dataloader,
                save_dir=os.path.join(self.test_dir, "test_resume"),
                save_per=4,
                cuda=[0])
        outname = os.path.join(self.test_dir, "test_pickle.trn")
        torch.save(self.trainer, outname)

    def test_prod_train(self):
        self.dataloader = ld.get_loader("vcoco_train", ld.COCO_IMGDIR)
        self.model = model.CtxBB()
        self.trainer = model.BasicTrainer(
                self.model, self.dataloader,
                save_dir=os.path.join(self.test_dir, "test_resume"),
                save_per=4,
                cuda=[0])
        outname = os.path.join(self.test_dir, "test_pickle.trn")
        self.trainer.train(1)

    def test_prod_test(self):
        self.model = model.TestCtxBB()
        evaluator = ev.Evaluator(cuda=[0])
        ev.do_eval(evaluator, self.model, "vcoco_val")
