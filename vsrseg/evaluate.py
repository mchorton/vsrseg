import torch.autograd as ag
import vsrl_utils as vu
import load_data as ld

from vsrl_eval import VCOCOeval

import utils.mylogger as logging

class Evaluator(object):
    def __init__(self, **kwargs):
        self.cuda = kwargs["cuda"]

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

def do_eval(evaluator, model, vcoco_set):
    results = evaluator.evaluate_model(
            model, "vcoco_val", ld.COCO_IMGDIR)
    outfile = os.path.join(cfg.save_dir, "evaluation.pkl")
    with open(outfile, "w") as f:
        pik.dump(results, f)
    vcocoeval = VCOCOeval(
            ld.get_vsrl_labels(vcoco_set),
            ld.COCO_VCOCO_ANN,
            ld.get_ids(vcoco_set))
    vcocoeval._do_eval(outfile, ovr_thresh=0.5)
