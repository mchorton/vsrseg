from os import path

import torch
import torch.utils.data as td
import torchvision.datasets as dsets
import torchvision.transforms as tt
import vsrl_utils as vu

import model as md

def get_imgid_2_vcoco_labels(vcoco_all, coco):
    """
    Get a dict from annotation id to vcoco image labels.
    """
    ret = {}
    for verb_dict in vcoco_all:
        verb_dict = vu.attach_gt_boxes(verb_dict, coco)
        action_name = verb_dict["action_name"]
        for i in xrange(len(verb_dict["image_id"])):
            img_id = verb_dict["image_id"][i][0]
            if img_id not in ret:
                ret[img_id] = {
                        "image_id": img_id,
                        "verbs": {},
                    }
            # Don't overwrite verb_dict while iterating.
            ret[img_id]["verbs"][action_name] = \
                    {
                        "role_object_id": verb_dict["role_object_id"][i],
                        "role_name": verb_dict["role_name"],
                        "label": verb_dict["label"][i],
                        "role_bbox": verb_dict["role_bbox"][i],
                        "include": verb_dict["include"],
                        "bbox": verb_dict["bbox"][i],
                    }
    return ret

class VCocoBoxes(dsets.coco.CocoDetection):
    """
    Subclass of CocoDetection dataset offered by pytorch's torchvision library
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py
    """
    def __init__(
            self, vcoco_set, root, transform=None, coco_transform=None, 
            combined_transform=None):
        # Don't call the superconstructor (we don't have an annFile)
        self.root = root
        self.coco = vu.load_coco()
        self.vcoco_all = vu.load_vcoco(vcoco_set)
        # If we don't convert to int, COCO library index lookup fails :(
        self.ids = [int(x) for x in self.vcoco_all[0]["image_id"].ravel()]
        self.transform = transform
        self.target_transform = coco_transform
        self.combined_transform = combined_transform

        # Get per-image vcoco labels, indexed by image id.
        self.imgid_2_vcoco = get_imgid_2_vcoco_labels(self.vcoco_all, self.coco)

    def __getitem__(self, index):
        img_id = self.ids[index]
        vcoco_ann = self.imgid_2_vcoco[img_id]
        img, coco_ann = super(VCocoBoxes, self).__getitem__(index)
        target = (coco_ann, vcoco_ann)
        if self.combined_transform is not None:
            target = self.combined_transform(target)
        return (img, target)

def get_loader(vcoco_set, coco_dir):
    transforms = tt.Compose([
            tt.Scale(md.IMSIZE),
            tt.ToTensor(),
        ])
    targ_trans = lambda y: torch.Tensor(y[1]["verbs"]["throw"]["label"])
    dataset = VCocoBoxes(
            vcoco_set, coco_dir, transform=transforms,
            combined_transform=targ_trans)
    return td.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

def get_label_loader(vcoco_set, coco_dir):
    transforms = tt.Compose([
            tt.Scale(md.IMSIZE),
            tt.ToTensor(),
        ])
    dataset = VCocoBoxes(
            vcoco_set, coco_dir, transform=transforms)
    return td.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
