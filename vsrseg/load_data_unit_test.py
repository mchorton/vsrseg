import unittest
import numpy as np

import load_data as ld

vcoco_all_test = [
        {
                "action_name": "action1",
                "role_name": ["agent", "obj"]},
        {
                "action_name": "action2",
                "role_name": ["agent", "instrument"]},
        {
                "action_name": "action3",
                "role_name": ["agent", "obj", "instrument"]},
    ]

vcoco_labels = {"verbs":
        {
                "action1": {"label": 1},
                "action2": {"label": 0},
                "action3": {"label": 1},
        }}

categories_test = ["person", "object1", "instrument1"]

class VCocoTranslatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = ld.VCocoTranslator(vcoco_all_test, categories_test)

    def test_human_object_gt_pairs(self):
        vcoco_labels
        action_labels = self.translator.get_action_labels(vcoco_labels)
        np.testing.assert_array_equal(
                np.array([1, 0, 1]),
                action_labels)

    # TODO it'd probably be good to add more tests
