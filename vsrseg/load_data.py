from os import path

import torch
import torch.utils.data as td
import torchvision.datasets as dsets
import torchvision.transforms as tt
import vsrl_utils as vu
from faster_rcnn.datasets.factory import get_imdb
import faster_rcnn.roi_data_layer.roidb as rdl_roidb
import numpy as np

import model as md

COCO_IMGDIR = "/home/mchorton/data/coco/images/"
VCOCO_DIR = "/home/mchorton/code/vsr_segmentation/v-coco/"

COCO_VCOCO_ANN = path.join(VCOCO_DIR, "data/instances_vcoco_all_2014.json")

def get_vsrl_labels(vcoco_set):
    return path.join(VCOCO_DIR, "data/vcoco/%s.json" % vcoco_set)

def get_ids(vcoco_set):
    return path.join(VCOCO_DIR, "data/splits/%s.ids" % vcoco_set)

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
                        #"image_path": coco.loadImgs([img_id])[0]["filename"],
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

def role_is_not_agent(agentrole_list):
    return "agent" != x[1]

def split_action_role(agentrole):
    return agentrole.split("-")

class VCocoTranslator(object):
    def __init__(self, vcoco_all, categories):
        self.ids_2_actions = sorted([x['action_name'] for x in vcoco_all])
        self.actions_2_ids = {
                name: i for i, name in enumerate(self.ids_2_actions)}

        # Create a local int-based mapping for nouns
        all_cats = ["__background__"] + sorted(categories)
        self.ids_2_nouns = all_cats
        self.nouns_2_ids = {x: i for i, x in enumerate(all_cats)}
        #classes = ['__background__'] + sorted([x['
        #self.ids_2_classes = sortedo

        # this is a mapping that combines verb with role for localization
        # tasks.
        actionroles = []
        actionroles_nonagent = []
        self.action_roles_lookup = {}
        for verb in vcoco_all:
            roles = verb["role_name"]
            self.action_roles_lookup[verb["action_name"]] = roles
            for role in roles:
                actionrole_name = "%s-%s" % (verb["action_name"], role)
                actionroles.append(actionrole_name)
                if role != "agent":
                    actionroles_nonagent.append(actionrole_name)
        self.ids_2_actionroles = sorted(actionroles)
        self.ids_2_actionrolesnonagent = sorted(actionroles_nonagent)
        self.actionroles_2_ids = {
                x: i for i, x in enumerate(self.ids_2_actionroles)}
        self.actionrolesnonagent_2_ids = {
                x: i for i, x in enumerate(self.ids_2_actionrolesnonagent)}

    @property
    def num_actions(self):
        return len(self.ids_2_actions)

    @property
    def num_action_roles(self):
        return len(self.ids_2_actionroles)

    @property
    def num_action_nonagent_roles(self):
        return len(self.ids_2_actionrolesnonagent)

    def get_action_labels(self, vcoco_labels):
        """
        Get numeric labels for v-coco action classes
        vcoco_labels: a dict like: {"verbs": {"verb_name": {"label": 0 or 1}}}
        """
        ret = np.empty(self.num_actions)
        for verb_name, labels in vcoco_labels["verbs"].iteritems():
            ret[self.actions_2_ids[verb_name]] = labels["label"]
        return ret

    def get_action_nonagent_role_locations(self, vcoco_labels):
        """
        Get a np.ndarray with size [1 x NActionRolesNonagent x 4]
        """
        ret = np.empty([1, self.num_action_nonagent_roles, 5], dtype=np.float)
        for index, actionrole in enumerate(self.ids_2_actionrolesnonagent):
            action, role = actionrole.split("-")
            position = vcoco_labels["verbs"][action]["role_name"].index(role)
            ret[0,index,1:] = self.get_nth_role_bbox(
                    vcoco_labels["verbs"][action]["role_bbox"], position)
            ret[0,index,0] = vcoco_labels["verbs"][action]["label"] * 1.
            print "==> index %d" % index
            print "vcoco_labels[verbs][%s]" % action, \
                    vcoco_labels["verbs"][action]
        return ret

    def action_role_iter(self):
        return it.ifilter(role_is_not_agent, it.imap(split_action_role, a))

    def get_nth_role_bbox(self, numpy_data, index):
        return numpy_data[(4*index):(4*(index + 1))]

    def get_human_object_gt_pairs(self, vcoco_labels):
        """
        TODO should a human-object pair only be trained for the single action
        on which its label appears?

        NBoxes will be the number of positive instances where a g.t. object and
        human have a positive label.
        Returns a tuple:
        tup[0] - a [NBoxes x 4] numpy.ndarray of human boxes
        tup[1] - a [NBoxes x 4] numpy.ndarray of object boxes
        tup[2] - a [NBoxes x NActionNonagentRoles] numpy.ndarray of gt labels

        It also ignores boxes in vcoco_labels that don't have a dimensions.
        """
        tup0 = []
        tup1 = []
        tup2 = []
        for index, actionrole in enumerate(self.ids_2_actionrolesnonagent):
            action, role = actionrole.split("-")
            if vcoco_labels["verbs"][action]["label"]:
                # This h_position quantity is always 0, AFAIK. Since agents are
                # always listed first.

                h_position = vcoco_labels["verbs"][action]["role_name"].index(
                        "agent")
                o_position = vcoco_labels["verbs"][action]["role_name"].index(
                        role)
                role_bbox = vcoco_labels["verbs"][action]["role_bbox"]
                if np.any(np.isnan(self.get_nth_role_bbox(
                        role_bbox, o_position))):
                    continue
                tup0.append(self.get_nth_role_bbox(role_bbox, h_position))
                tup1.append(self.get_nth_role_bbox(role_bbox, o_position))

                gt_labels = np.zeros(self.num_action_nonagent_roles)
                gt_labels[index] = 1.
                tup2.append(gt_labels)
        return map(np.vstack, [tup0, tup1, tup2])

    def human_scores_to_agentrolenonagent(self, h_scores):
        # Make something that is [NxNActions] into something that puts those
        # action scores only in locations corresponding to action-nonagent
        # prediction slots.
        ret = np.empty([h_scores.shape[0], self.num_action_nonagent_roles])
        for index, action in enumerate(self.ids_2_actions, start=0):
            roles = self.action_roles_lookup[action]
            for role in roles:
                if role == "agent":
                    continue
                actionrole = "%s-%s" % (action, role)
                ret_ind = self.actionrolesnonagent_2_ids[actionrole]
                ret[:, ret_ind] = h_scores[:, index]
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

class RoiVCocoBoxes(VCocoBoxes):
    """
    Subclass of CocoDetection dataset offered by pytorch's torchvision library
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py
    """
    def __init__(
            self, vcoco_set, root):
        super(RoiVCocoBoxes, self).__init__(vcoco_set, root)

        if vcoco_set == "vcoco_train":
            coco_split = "train"
        elif vcoco_set == "vcoco_val":
            coco_split = "val"
        else:
            raise ValueError("Invalid vcoco_set '%s'" % vcoco_set)
        imdb_name = "coco_2014_" + coco_split
        self._imdb = get_imdb(imdb_name)
        rdl_roidb.prepare_roidb(self._imdb)
        self._roidb = self._imdb.roidb

        self.cocoimgid_2_roidbindex = {
                index: i for i, index in enumerate(self._imdb._image_index)}

    def __getitem__(self, index):
        img_id = self.ids[index]
        vcoco_ann = self.imgid_2_vcoco[img_id]
        roidb_entry = self._roidb[self.cocoimgid_2_roidbindex[img_id]]
        return (roidb_entry, vcoco_ann)

def targ_trans(target):
    return torch.Tensor(target[1]["verbs"]["throw"]["label"])

# TODO delete this.
def get_loader(vcoco_set, coco_dir):
    transforms = tt.Compose([
            tt.Scale(md.IMSIZE),
            tt.ToTensor(),
        ])
    dataset = VCocoBoxes(
            vcoco_set, coco_dir, transform=transforms,
            combined_transform=targ_trans)
    return td.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

def get_label_loader(vcoco_set, coco_dir, test=False):
    # Similar to get_loader(), but gives a loader that gives all the full labels
    transforms = tt.Compose([
            tt.Scale(md.IMSIZE),
            tt.ToTensor(),
        ])
    if not test:
        cls = VCocoBoxes
    else:
        cls = FakeVCocoBoxes
    dataset = cls(vcoco_set, coco_dir, transform=transforms)
    return td.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

class FakeVCocoBoxes(VCocoBoxes):
    def __len__(self):
        return 40

class FakeDatasetLoader(object):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)

def make_vcoco_test_loader():
    loader = get_label_loader("vcoco_train", COCO_IMGDIR)
    outloc = "data/test_data.th"
    make_test_loader(loader, outloc)

def get_test_dataset(loader):
    items = []
    for i, data in enumerate(loader):
        items.append(data)
        if i > 0:
            break
    dataset = FakeDatasetLoader(items)
    return dataset

def make_test_loader(loader, outloc):
    dataset = get_test_dataset(loader)
    torch.save(dataset, outloc)

def get_test_loader():
    dataset = torch.load("data/test_data.th")
    return dataset

def make_roi_test_loader():
    loader = RoiVCocoBoxes("vcoco_train", COCO_IMGDIR)
    outloc = "data/test_roi_data.th"
    dataset = get_test_dataset(loader)
    torch.save((dataset, loader._imdb._classes), outloc)

def get_roi_test_loader():
    dataset, classes = torch.load("data/test_roi_data.th")
    return dataset, classes
