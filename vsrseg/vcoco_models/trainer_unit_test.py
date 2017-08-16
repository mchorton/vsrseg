import unittest

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
        categories = [x["name"] for x in vu.load_coco().cats.itervalues()]
        translator = ld.VCocoTranslator(vcoco_all, categories)
        n_action_classes = translator.num_actions
        n_action_nonagent_roles = translator.num_action_nonagent_roles
        cls.model = fhoi.HoiModel(
                cls.classes, n_action_classes, n_action_nonagent_roles,
                faster_rcnn_command_line=["NCLASSES", len(cls.classes)],
                cuda=[0])
        cls.loss = thoi.HoiLoss(cls.model, logger_output=cls.logger)

    def test_loss_forward(self):
        for x,y in self.roi_dataset:
            loss = self.loss([x], y)
            self.assertEqual(0, loss)
