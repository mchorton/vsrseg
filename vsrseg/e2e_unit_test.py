import unittest
import tempfile
import shutil
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import model as md
import vsrseg.load_data as ld

class E2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.model = md.TestCtxBB()
        #self.dataloader = ld.get_label_loader(
        #        "vcoco_train", ld.COCO_IMGDIR, test=True)
        self.dataloader = ld.get_test_loader()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_resume_train(self):
        self.trainer = md.TestTrainerRealData(
                self.model, self.dataloader, cuda=[], 
                save_dir=os.path.join(self.test_dir, "test_resume"),
                save_per=4)
        self.trainer.train(2)
        self.assertEqual(self.trainer.epoch, 2)
