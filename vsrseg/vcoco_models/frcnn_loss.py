"""
Based on faster_rcnn's build_loss functions. Saving local state to the
FasterRCNN can cause issues with parallelism in PyTorch (namely, forward()
cannot save local state when using DataParallel). This code was created to
circumvent such changes to local state.
"""

# TODO this code seems entirely unused.

import torch
import torch.nn.functional as F

def rpn_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
    # classification loss
    rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    rpn_label = rpn_data[0].view(-1)

    rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

    fg_cnt = torch.sum(rpn_label.data.ne(0))

    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    # box loss
    rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
    rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
    rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

    rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

    return rpn_cross_entropy, rpn_loss_box

def faster_rcnn_loss(cls_score, bbox_pred, roi_data):
    # classification loss
    label = roi_data[1].squeeze()
    fg_cnt = torch.sum(label.data.ne(0))
    bg_cnt = label.data.numel() - fg_cnt

    ce_weights = torch.ones(cls_score.size()[1])
    ce_weights[0] = float(fg_cnt) / bg_cnt
    ce_weights = ce_weights.cuda()
    cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

    # bounding box regression L1 loss
    bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
    bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
    bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

    loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

    return cross_entropy, loss_box
