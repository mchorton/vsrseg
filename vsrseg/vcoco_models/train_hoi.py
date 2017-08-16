# TODO I'm pretty sure the predictions for a multivalued role are just treated
# independently. Make sure.
# TODO pretrain Faster RCNN on COCO
# TODO freeze the RPN after pretraining?
# TODO BGR order expected for images 
#       (see faster_rcnn_pytorch/faster_rcnn/utils/blob.py)
# TODO weight initialization?

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from faster_rcnn.roi_data_layer.minibatch import get_minibatch
import faster_rcnn.network as network
import vsrl_utils as vu

import vsrseg.model as md
from frcnn_loss import faster_rcnn_loss, rpn_loss
from fair_hoi import faster_rcnn_forward
import vsrseg.load_data as ld


# TODO be careful when parallelizing this part. We're changing local state
# of the Trainer.
class HoiLoss(nn.Module):
    def __init__(self, model, logger_output=None):
        super(HoiLoss, self).__init__()
        self.logger = logger_output

        vcoco_all = vu.load_vcoco("vcoco_train")
        categories = [x["name"] for x in vu.load_coco().cats.itervalues()]
        self.vcoco_translator = ld.VCocoTranslator(vcoco_all, categories)

        self.model = model

    def log_values(self, dict_):
        if self.logger:
            self.logger.append(dict_)

    def forward(self, roidb, vcoco_ann):
        """
        Get a {"name": Loss} mapping from a given x,y datapoint.
        The losses will later be summed, but it's convenient to store them
        individually for logging purposes.
        """
        ret = {}
        blobs = get_minibatch(roidb, len(self.model.detection_branch.classes))

        im_data = blobs['data']
        im_info = blobs['im_info']
        gt_boxes = blobs['gt_boxes']
        gt_ishard = blobs['gt_ishard']
        dontcare_areas = blobs['dontcare_areas']

        # Get cross-entropy and box loss for rpn and faster-rcnn networks
        cls_prob, bbox_pred, rois, features, rpn_ce, rpn_lb, f_ce, f_lb = \
                faster_rcnn_forward(
                        self.model.detection_branch, im_data, im_info, gt_boxes,
                        gt_ishard, dontcare_areas)
        ret.update({
                "rpn_ce": rpn_ce,
                "rpn_lb": rpn_lb,
                "f_ce": f_ce,
                "f_lb": f_lb})

        # Find human boxes that have >= 0.5 overlap with gt
        # RB has elements; want rb[0]['gt_boxes']
        # TODO these person_indexes are empty (?)
        person_index = self.vcoco_translator.nouns_2_ids["person"]
        assert len(roidb) == 1, "Invalid len(roidb) > 1" # This code requires it
        elem = roidb[0]
        candidate_persons = np.where(np.logical_and(
                elem["gt_classes"] == person_index,
                elem["gt_overlaps"][:, person_index] > 0.5))
        # TODO b_h is empty...
        # TODO this data that we're feeding is WAY wrong. the filename
        # corresponds to a picture of a surfer, labels show airplanes...
        b_h = elem["boxes"][candidate_persons]
        np.random.shuffle(b_h)
        b_h = b_h[:16] # only choose 16 boxes.
        b_h = np.array([[1., 1., 2., 2.]]) # TODO :/
        b_h = network.np_to_variable(b_h)

        action_scores, action_locations = self.model.human_centric_branch(
                b_h, features)

        # Get ground-truth and calculate loss.
        # This is [B=1 x NActions]
        gt_action_scores = self.vcoco_translator.get_action_labels(vcoco_ann)
        gt_action_scores = network.np_to_variable(gt_action_scores)

        # it will be a [B=1 x NActionNonagentRoles x 5] structure]
        # and the actions for 2-obj action things are treated uniquely.
        # I removed 'Nan's also, appear to be unlabeled.
        gt_action_locations = \
                self.vcoco_translator.get_action_nonagent_role_locations(
                        vcoco_ann)
        chosen_locations = np.where(np.logical_and(
                gt_action_locations[:, :, 0] == 1,
                np.logical_not(np.isnan(gt_action_locations[:, :, 1]))))
        gt_action_locations = gt_action_locations[chosen_locations]
        gt_action_locations = network.np_to_variable(gt_action_locations)

        action_locations = action_locations.cpu().data.numpy()
        action_locations = action_locations[chosen_locations]
        action_locations = network.np_to_variable(action_locations)

        action_ce = F.binary_cross_entropy(action_scores, gt_action_scores)
        location_l1 = F.smooth_l1_loss(
                action_locations, gt_action_locations[:, 1:])

        ret.update({
                "action_ce": action_ce,
                "location_l1": location_l1})

        # TODO the last part is confusing. I'll take it to mean that b_h and b_o
        # must both be taken from ground truth labels.
        # (But possibly, they mean that only the cases where the action has a
        # positive label for those boxes)
        # Get the gT human box.
        # TODO consider removing this part of the system...?
        person_gt_boxes_loc = np.where(elem["gt_classes"] == person_index)
        person_gt_boxes = elem["gt_classes"][person_gt_boxes_loc]
        # TODO this will give another gradient to human branch... ???
        # gt_actions will probably be one-hot along each row for those
        # interactions? Or not...; can just expand gt_action_scores from above.
        b_h, b_o, gt_actions = self.vcoco_translator.get_human_object_gt_pairs(
                vcoco_ann)
        b_h, b_o, gt_actions = map(
                network.np_to_variable, [b_h, b_o, gt_actions])
        h_action_scores, _ = self.model.human_centric_branch(
                b_h, features)
        h_action_scores = \
                self.vcoco_translator.human_scores_to_agentrolenonagent(
                        h_action_scores.cpu().data.numpy())
        h_action_scores = network.np_to_variable(h_action_scores)
        scores = self.model.interaction_branch(h_action_scores, b_o, features)

        interaction_ce = F.binary_cross_entropy(scores, gt_actions)

        ret.update({"interaction_ce": interaction_ce})

        self.log_values(ret)

        loss = sum(ret.itervalues())
        return loss

class HoiTrainer(md.BasicTrainer):
    def __init__(self, model, dataloader, **kwargs):
        super(HoiTrainer, self).__init__(model, dataloader, **kwargs)

        # Create the loss function, give it the epoch_data to log to.
        self.loss = HoiLoss(model, self.epoch_data)

        # TODO these losses are based on faster_rcnn codebase, not mentioned
        # in HOI paper.
        # TODO will these optimizers all be fighting each other? Do I need to
        # prevent propagation of gradients all the way back? Should I just use
        # one optimizer?
        # TODO freeze weights for faster RCNN??
        self.optimizer = torch.optim.SGD(lr=self.lr)
        # TODO why params[8:]? (Taken from original FRCNN training code.)
        #faster_rcnn_optimizer = torch.optim.SGD(
        #       params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    def handle_batch(self, data):
        # TODO figure out how to best parallelize this
        x, y = data
        x = ag.Variable(x)
        y = ag.Variable(y)
        if self.cuda:
            x = x.cuda(self.cuda[0])
            y = y.cuda(self.cuda[0])

        self.optimizer.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        self.optimizer.step()
