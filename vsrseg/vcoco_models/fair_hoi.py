"""
Reimplementation of models from https://arxiv.org/pdf/1704.07333.pdf
"""

import torch.nn as nn
import faster_rcnn.fast_rcnn.config as cf
import faster_rcnn.network as network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.roi_pooling.modules.roi_pool import RoIPool
import torch.nn.functional as F

import vsrseg.model as md

class HumanCentricBranch(nn.Module):
    def __init__(
            self, n_action_classes, n_action_nonagent_roles, pool_size=7,
            in_filters=512, **kwargs):
        super(HumanCentricBranch, self).__init__()
        # TODO should it have its own pool layer, or reuse that of the detector?
        self.roi_pool = RoIPool(pool_size, pool_size, 1.0/16)
        pool_outdim = in_filters * (pool_size ** 2)
        self.feat_2_scores = nn.Linear(pool_outdim, n_action_classes)
        self.n_action_nonagent_roles = n_action_nonagent_roles
        # Note: dimensionality of the output target localizations is 4
        # TODO should be 4 * number of actions.
        # TODO this model is kinda weird and probably naive.
        self.feat_2_locations = nn.Sequential(
                nn.Linear(pool_outdim, pool_outdim),
                nn.ReLU(),
                nn.Linear(pool_outdim, 4 * n_action_nonagent_roles))

    # TODO shift coordinate frame of these predictions??
    def forward(self, b_h, features):
        """
        b_h: a [n_bboxes X bbox_dim] tensor with human-centric bounding boxes
        Be sure to only call this on human boxes!
        TODO should I use a new ROI pooling layer?

        Returns:
        - scores: a [n_bboxes X n_action_nonagent_roles] np.ndarray of scores
        - locations: a [n_bboxes X n_action_nonagent_roles X 4] np.ndarray
                of locations
        """
        self.cuda() # TODO why is none of model on cuda? Did I bork up?
        pooled_features = self.roi_pool(features, b_h)
        # TODO I assume this is what the paper meant by binary sigmoid
        # classifiers
        aligned_features = pooled_features.view(b_h.size(0), -1)
        scores = F.sigmoid(self.feat_2_scores(aligned_features))
        # TODO not 100% certain about this. We probably want some nonlinearity?
        locations = self.feat_2_locations(aligned_features)
        locations = locations.view(b_h.size(0), self.n_action_nonagent_roles, 4)
        return scores, locations
        # TODO my return for this thing should probably just be one single big
        # structure.

class InteractionBranch(nn.Module):
    # TODO do we also make a new pooling layer here?
    def __init__(
            self, n_action_nonagent_roles, in_filters=512, pool_size=7,
            **kwargs):
        super(InteractionBranch, self).__init__()
        self.roi_pool = RoIPool(pool_size, pool_size, 1.0/16)
        pool_outdim = in_filters * (pool_size ** 2)
        self.feat_2_scores = nn.Linear(pool_outdim, n_action_nonagent_roles)

    # TODO are we supposed to use the actual output scores of HumanCentric
    # branch, or some intermediate computation?
    def forward(self, h_scores, b_o, features):
        """
        h_scores: a [n_bboxes X n_actions] tensor with human-centric action
                  probabilities
        b_o: a [n_bboxes X bbox_dim] tensor with object bounding boxes
        features: the features to be pooled

        returns: 
        """
        # TODO make sure this is similar to the HumanCentric forward() 
        self.cuda() # TODO why is none of model on cuda? Did I bork up?
        pooled_features = self.roi_pool(features, b_o)
        aligned_features = pooled_features.view(b_o.size(0), -1)
        o_scores = F.sigmoid(self.feat_2_scores(aligned_features))

        assert h_scores.size() == o_scores.size(), \
                "h_scores.size()=%s doesn't match o_scores.size()=%s" % \
                        (str(h_scores.size()), str(o_scores.size()))
        
        final_scores = F.sigmoid(h_scores + o_scores)
        return final_scores

def rpn_forward(
        rpn, im_data, im_info, gt_boxes=None, gt_ishard=None,
        dontcare_areas=None):
    im_data = network.np_to_variable(im_data, is_cuda=True)
    im_data = im_data.permute(0, 3, 1, 2)
    # TODO not sure how to best handle this (with paralellism)
    rpn.cuda(0) # TODO needed for tensors to be on same gpus, somehow
    features = rpn.features(im_data)

    rpn_conv1 = rpn.conv1(features)

    # rpn score
    rpn_cls_score = rpn.score_conv(rpn_conv1)
    rpn_cls_score_reshape = rpn.reshape_layer(rpn_cls_score, 2)
    rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
    rpn_cls_prob_reshape = rpn.reshape_layer(
            rpn_cls_prob, len(rpn.anchor_scales)*3*2)

    # rpn boxes
    rpn_bbox_pred = rpn.bbox_conv(rpn_conv1)

    # proposal layer
    cfg_key = 'TRAIN' if rpn.training else 'TEST'
    rois = rpn.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                               cfg_key, rpn._feat_stride, rpn.anchor_scales)

    ret = [features, rois]

    # generating training labels and build the rpn loss
    if rpn.training:
        assert gt_boxes is not None
        rpn_data = rpn.anchor_target_layer(
                rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info,
                rpn._feat_stride, rpn.anchor_scales)
        cross_entropy, loss_box = rpn.build_loss(
                rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
        ret += [cross_entropy, loss_box]

    return ret

def faster_rcnn_forward(
        faster_rcnn, im_data, im_info, gt_boxes=None, gt_ishard=None,
        dontcare_areas=None):
    """
    Reimplementation of Faster RCNN's forward() function. It behaves slightly
    differently, and returns slightly different information
    """
    rpn_output = rpn_forward(
            faster_rcnn.rpn, im_data, im_info, gt_boxes, gt_ishard,
            dontcare_areas)
    features, rois = rpn_output[0], rpn_output[1]

    # TODO should I def do this?
    if faster_rcnn.training:
        roi_data = faster_rcnn.proposal_target_layer(
                rois, gt_boxes, gt_ishard, dontcare_areas,
                faster_rcnn.n_classes)
        rois = roi_data[0]

    # roi pool
    pooled_features = faster_rcnn.roi_pool(features, rois)
    x = pooled_features.view(pooled_features.size()[0], -1)
    faster_rcnn = faster_rcnn.cuda(0) # TODO hmm...
    x = faster_rcnn.fc6(x)
    x = F.dropout(x, training=faster_rcnn.training)
    x = faster_rcnn.fc7(x)
    x = F.dropout(x, training=faster_rcnn.training)

    cls_score = faster_rcnn.score_fc(x)
    cls_prob = F.softmax(cls_score)
    bbox_pred = faster_rcnn.bbox_fc(x)

    ret = [cls_prob, bbox_pred, rois, features]

    if faster_rcnn.training:
        cross_entropy, loss_box = faster_rcnn.build_loss(
                cls_score, bbox_pred, roi_data)
        rpn_cross_entropy, rpn_loss_box = rpn_output[2], rpn_output[3]
        ret += [
                rpn_cross_entropy,
                rpn_loss_box,
                cross_entropy,
                loss_box,
                roi_data,
            ]
    return ret

def human_object_boxes(bbox_pred, cls_prob, class_indices):
    """
    Split bbox_pred into a tensor of human boxes and a tensor of object boxes.
    """
    # TODO is it possible to do this on the GPU?
    scores, inds = cls_prob.data.max(1)
    scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
    class_names = class_indices[inds]
    h_inds = np.where(class_names == "person")
    o_inds = np.where(class_names != "person")

    b_h = bbox_pred[h_inds]
    b_o = bbox_pred[o_inds]
    return b_h, b_o

# Note that, FasterRCNN (used by this) has a global config :( , so trying to 
# create multiple HoiModels with different FasterRCNN configs will cause issues.

#class FakeHoiModel(nn.Module):


class HoiModel(nn.Module):
    def __init__(
            self, classes, n_action_classes, n_action_nonagent_roles, **kwargs):
        super(HoiModel, self).__init__()
        print "Constructing HOI Model"

        faster_rcnn_config = kwargs.get("faster_rcnn_config", None)
        if faster_rcnn_config is not None:
            cf.cfg_from_file(faster_rcnn_config)

        faster_rcnn_cle = kwargs.get("faster_rcnn_command_line", None)
        if faster_rcnn_cle is not None:
            cf.cfg_from_list(faster_rcnn_cle)

        assert(cf.cfg["NCLASSES"] == len(classes)), \
                "inconsistent FasterRCNN settings"

        self.detection_branch = FasterRCNN(classes=classes)

        self.human_centric_branch = HumanCentricBranch(
                n_action_classes, n_action_nonagent_roles)
        self.interaction_branch = InteractionBranch(n_action_nonagent_roles)

'''
Need:
- classes from VCOCO
- well-formatted im_info input for the detector
- maybe just sub-class the FasterRCNN implementation. Or just copy/rewrite the
  code. remove the build_loss, etc.


It looks like the RPN takes in only a single image at a time (?). :( 
- multigpu might be painful :(

Unfortunately, the proposal_layer requires a hand-laid config. I need to change
this.
'''
