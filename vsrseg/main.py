import sys
import argparse

import torch
import torch.utils.data as td

import model as md
import utils.mylogger as logging

def get_loader():
    # TODO actually get real data. For now, hallucinate a dataset of all-0 imgs
    dataset = td.TensorDataset(torch.zeros(32, 256, 256, 3), torch.ones(32, 4))
    dataloader = td.DataLoader(
            dataset, batch_size=8, shuffle=True, num_workers=0)
    return dataloader

def main(args):
    parser = argparse.ArgumentParser(
            description=(
                    "Run deep models for visual semantic role segmentation "
                    "(or detection)"))
    parser.add_argument("mode", help="Mode to run model in (e.g. 'train')")
    cfg = parser.parse_args(args)

    if cfg.mode == 'train':
        model = md.CtxBB()
        dataloader = get_loader()
        trainer = md.BBTrainer(model, dataloader)
        trainer.train(50)
    else:
        logging.getLogger(__name__).error("Invalid mode '%s'" % str(cfg.mode))
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
