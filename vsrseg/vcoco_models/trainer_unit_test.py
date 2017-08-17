import unittest
import numpy as np

from faster_rcnn.roi_data_layer.minibatch import get_minibatch

import train_hoi as thoi
import fair_hoi as fhoi
import vsrseg.load_data as ld
import vsrl_utils as vu

class LossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #cls.test_dir = tempfile.mkdtemp()
        cls.roi_dataset, cls.classes = ld.get_roi_test_loader()
        # This might include non v-coco classes, but is ok for testing.
        cls.logger = []
        # TODO this is known!

        vcoco_all = vu.load_vcoco("vcoco_train")

        # give the fake dataloader stuff that the RoiDataset needs.
        cls.roi_dataset.vcoco_all = vcoco_all
        cls.roi_dataset.get_classes = lambda : cls.classes

        categories = [x["name"] for x in vu.load_coco().cats.itervalues()]
        cls.translator = ld.VCocoTranslator(vcoco_all, categories)
        n_action_classes = cls.translator.num_actions
        n_action_nonagent_roles = cls.translator.num_action_nonagent_roles
        cls.model = fhoi.HoiModel(
                cls.classes, n_action_classes, n_action_nonagent_roles,
                faster_rcnn_command_line=["NCLASSES", len(cls.classes)],
                cuda=[0])
        cls.loss = thoi.HoiLoss(
                cls.model, cls.translator, logger_output=cls.logger)
        cls.trainer = thoi.HoiTrainer(cls.model, cls.roi_dataset, cuda=[0])

    # TODO wait why does this pass? should assert false...
    def test_loss_forward(self):
        for x,y in self.roi_dataset:
            loss = self.loss([x], y)
            self.assertEqual(0, loss)

    def test_train_loop(self):
        self.trainer.train(2)

    def test_visualize_minibatch(self):
        for x,vcoco_ann in self.roi_dataset:
            roidb = [x]
            blobs = get_minibatch(
                    roidb, vcoco_ann)
            im_data = blobs['data']
            cls_label = 0
            vis = thoi.Visualizer(self.translator)
            roi = 10 * np.arange(5)
            vis.vis_sample(roi, im_data, cls_label)

    def test_visualize_samples(self):
        for x,vcoco_ann in self.roi_dataset:
            roidb = [x]
            blobs = get_minibatch(
                    roidb, vcoco_ann)
            im_data = blobs['data']
            cls_label = 0
            vis = thoi.Visualizer(self.translator)
            rois = 10 * np.vstack((np.arange(4), np.arange(4) + 30))
            gt_classes = np.array([2, 5])
            gt_overlaps = np.random.rand(2, len(self.classes))

            vis.visualize_samples(im_data, gt_classes, gt_overlaps, rois)
