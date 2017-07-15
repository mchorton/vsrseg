import sys
import argparse

import torch
import torch.utils.data as td
import torchvision.transforms as tt

import model as md
import utils.mylogger as logging
import load_data as ld

def get_loader(vcoco_set, coco_dir):
    # TODO need to update to get tt.Scale class.
    transforms = tt.Compose([
            tt.Scale(md.IMSIZE),
            tt.ToTensor(),
        ])
    targ_trans = lambda y: torch.Tensor(y[1]["throw"]["label"])
    dataset = ld.VCocoBoxes(
            vcoco_set, coco_dir, transform=transforms,
            combined_transform=targ_trans)
    return td.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

def main(args):
    parser = argparse.ArgumentParser(
            description=(
                    "Run deep models for visual semantic role segmentation "
                    "(or detection)"))
    parser.add_argument("mode", help="Mode to run model in (e.g. 'train')")
    parser.add_argument(
            "-e", "--epochs", help="number of epochs for training", default=50)
    parser.add_argument(
            "-l", "--learn_rate", help="learning rate", type=float,
            default=0.001)
    parser.add_argument(
            "-c", "--cpu", action="store_true",
            help="flag to use cpu instead of cuda")
    cfg = parser.parse_args(args)

    if cfg.mode == 'train':
        model = md.CtxBB()
        dataloader = get_loader(
                "vcoco_train", "../v-coco/coco/images/train2014")
        trainer = md.BBTrainer(
                model, dataloader, cuda=(not cfg.cpu), lr=cfg.learn_rate)
        trainer.train(cfg.epochs)
    else:
        logging.getLogger(__name__).error("Invalid mode '%s'" % str(cfg.mode))
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
