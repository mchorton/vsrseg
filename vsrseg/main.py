import sys
import argparse
import datetime as dt
import os
import cPickle as pik

import torch
import torch.utils.data as td
import torchvision.transforms as tt

import model as md
import utils.mylogger as logging
import load_data as ld
import evaluate as ev

def get_fake_loader():
    datapoints = 64
    # First dim is batch; 3 channels, of size md.IMSIZE[0] x md.IMSIZE[1]
    x = torch.zeros(datapoints, 3, *md.IMSIZE)
    y = torch.ones(datapoints, 1)
    dataset = td.TensorDataset(x, y)
    return td.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

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
            "-f", "--fake", action="store_true",
            help=(
                    "flag to use fake data that loads quickly (for"
                    "development purposes)"))
    parser.add_argument(
            "--net", help="file in which model is stored. Used in test mode.",
            default=None)
    cfg = parser.parse_args(args)

    if cfg.mode == 'train':
        model = md.CtxBB()
        if cfg.fake:
            dataloader = get_fake_loader()
        else:
            dataloader = ld.get_loader(
                    "vcoco_train", ld.COCO_IMGDIR)
        trainer = md.BasicTrainer(model, dataloader, **vars(cfg))
        logging.getLogger(__name__).info("Beginning Training...")
        trainer.train(cfg.epochs)
    elif cfg.mode == 'test':
        checkpoint = torch.load(cfg.net)
        model = checkpoint["model"]
        evaluator = ev.Evaluator(**vars(cfg))
        ev.do_eval(evaluator, model, "vcoco_val")

    else:
        logging.getLogger(__name__).error("Invalid mode '%s'" % str(cfg.mode))
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
