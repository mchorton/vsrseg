"""
File for running the VCOCO experiment.
"""
import sys
import argparse
import datetime as dt
import os

import torch
import torch.utils.data as td
import torchvision.transforms as tt
import vsrl_utils as vu

import utils.mylogger as logging
import vsrseg.load_data as ld
import fair_hoi as fhoi
import train_hoi as thoi
#import evaluate as ev

def main(args):
    parser = argparse.ArgumentParser(
            description=(
                    "Run deep models for visual semantic role segmentation "
                    "(or detection)"))
    parser.add_argument("mode", help="Mode to run model in (e.g. 'train')")
    parser.add_argument(
            "-s", "--save_dir", help="directory for saving the model",
            default="saved_models/%s" % dt.datetime.now().strftime(
                    "%Y_%m_%d_%H_%M_%S"))
    parser.add_argument(
            "-e", "--epochs", help="number of epochs for training", type=int,
            default=50)
    parser.add_argument(
            "-p", "--save_per", help="epochs to wait before saving", type=int,
            default=5)
    parser.add_argument(
            "-l", "--learn_rate", help="learning rate", type=float,
            default=0.001)
    parser.add_argument(
            "-c", "--cuda", type=int, nargs="+",
            help="ids of gpus to use during training", default=[])
    parser.add_argument(
            "--net", help="file in which model is stored. Used in test mode.",
            default=None)
    parser.add_argument(
            "--coco_root", help=(
                    "path to coco directory, which must have an 'images'"
                    "subfolder with images inside"),
            default=ld.COCO_ROOT)
    parser.add_argument(
            "--vcoco_root", help=(
                    "path to v-coco directory, which must have a 'coco'"
                    "subfolder with another subfolder containing"
                    "images/train2014 and images/val2014"),
            default=ld.VCOCO_ROOT)
    cfg = parser.parse_args(args)

    pathman = ld.PathManager(coco_root=cfg.coco_root, vcoco_root=cfg.vcoco_root)

    if cfg.mode == 'train':
        vcoco_all = vu.load_vcoco("vcoco_train")
        categories = [x["name"] for x in vu.load_coco().cats.itervalues()]
        translator = ld.VCocoTranslator(vcoco_all, categories)
        n_action_classes = translator.num_actions
        n_action_nonagent_roles = translator.num_action_nonagent_roles
        dataloader = ld.RoiVCocoBoxes(
                "vcoco_train", pathman.coco_root, pathman.vcoco_root)
        classes = dataloader.get_classes()
        model = fhoi.HoiModel(
                classes, n_action_classes, n_action_nonagent_roles,
                faster_rcnn_command_line=["NCLASSES", len(classes)],
                cuda=cfg.cuda)
        trainer = thoi.HoiTrainer(model, dataloader, **vars(cfg))
        logging.getLogger(__name__).info("Beginning Training...")
        trainer.train(cfg.epochs)
        """
    # TODO build this test code.
    elif cfg.mode == 'test':
        checkpoint = torch.load(cfg.net)
        model = checkpoint["model"]
        evaluator = ev.Evaluator(**vars(cfg))
        ev.do_eval(evaluator, model, "vcoco_val", cfg.save_dir)
        """
    else:
        logging.getLogger(__name__).error("Invalid mode '%s'" % str(cfg.mode))
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
