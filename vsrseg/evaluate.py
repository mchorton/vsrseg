import torch.autograd as ag
import vsrl_utils as vu
import load_data as ld

import utils.mylogger as logging

class Evaluator(object):
    def __init__(self, **kwargs):
        self.cuda = not kwargs["cpu"]

    def evaluate_model(self, model, vcoco_set, coco_dir):
        if self.cuda:
            model = model.cuda()
        loader = ld.get_label_loader(vcoco_set, coco_dir)
        results = []
        for i, data in enumerate(loader):
            x, y = data
            x = ag.Variable(x)
            if self.cuda:
                x = x.cuda()
            pred = model(x)
            results += model.interpret_predictions(pred, y)
        return results
