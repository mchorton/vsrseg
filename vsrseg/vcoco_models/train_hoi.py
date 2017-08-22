# TODO I'm pretty sure the predictions for a multivalued role are just treated
# independently. Make sure.
# TODO pretrain Faster RCNN on COCO
# TODO freeze the RPN after pretraining?
# TODO BGR order expected for images 
#       (see faster_rcnn_pytorch/faster_rcnn/utils/blob.py)
# TODO weight initialization?

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from faster_rcnn.roi_data_layer.minibatch import get_minibatch
from faster_rcnn.fast_rcnn.config import cfg
import faster_rcnn.network as network
import vsrl_utils as vu
import tensorflow as tf

import vsrseg.model as md
from frcnn_loss import faster_rcnn_loss, rpn_loss
from fair_hoi import faster_rcnn_forward
import vsrseg.load_data as ld
import vsrseg.utils.mylogger as logging
import vsrseg.utils.methods as mt

def print_batch_loss(trainer):
    if len(trainer.epoch_data) == 0:
        return
    logging.getLogger(__name__).info(
            "---> After Epoch #%.5f: STATS:\n%s\n--> END STATS" % (
                    trainer.fractional_epoch(),
                    str(trainer.epoch_data[-1])))

def _log_error(trainer, prefix, timestep):
    # log error values
    if len(trainer.epoch_data) < 1:
        return
    if mt.should_do(trainer.iteration, 1) and (trainer.save_dir is not None):
        print "Recording stuff at timestep ", timestep
        err_dict = trainer.epoch_data[-1]
        for name, err in err_dict.iteritems():
            name = "%s%s" % (prefix, name)
            values = [
                    tf.Summary.Value(tag=name, simple_value=err),
                ]
            summary = tf.Summary(value=values)
            trainer.train_writer.add_summary(summary, timestep)

def epoch_log_error(trainer):
    print "logging epoch error"
    _log_error(trainer, "iter_", trainer.epoch)

def iter_log_error(trainer):
    print "logging iter error"
    _log_error(trainer, "", trainer.total_iterations())
# TODO be careful when parallelizing this part. We're changing local state
# of the Trainer.
class HoiLoss(nn.Module):
    def __init__(self, model, vcoco_translator, logger_output=None):
        super(HoiLoss, self).__init__()
        self.logger = logger_output

        self.vcoco_translator = vcoco_translator

        self.model = model
        self.visualizer = Visualizer(self.vcoco_translator)

    def log_values(self, dict_):
        if self.logger is not None:
            next_dict = {key: variable.data[0] for key, variable in
                    dict_.iteritems()}
            next_dict["sum"] = sum(next_dict.itervalues())
            self.logger.append(next_dict)
            

    # TODO ok, this is probably mostly right. Next steps:
    # 1. Get it to show losses in tensorboard.
    # 2. Get a pretrained network. (In future, self-train it.)
    # 3. Do inference and evaluation.
    def forward(self, roidb, vcoco_ann):
        # TODO freeze the RPN? If so, why?
        # TODO what is purpose of the non-gt roi stuff?
        # TODO do I need to change this to use non-gt-rois??? probably.
        # Also note that the system in HOI paper trained on both RPN proposals
        # and GT proposals. We're training on GT and some random stuff... ?
        # Let's punt for now.
        assert len(roidb) == 1, "Invalid len(roidb) > 1" # This code requires it
        assert cfg.TRAIN.FG_FRACTION == 0.25
        assert cfg.TRAIN.FG_THRESH == 0.5
        assert cfg.TRAIN.HAS_RPN, "Training this model requires an RPN"
        """
        Get a {"name": Loss} mapping from a given x,y datapoint.
        The losses will later be summed, but it's convenient to store them
        individually for logging purposes.
        """
        ret = {}
        blobs = get_minibatch(roidb, len(self.model.detection_branch.classes))
        #def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):

        im_data = blobs['data']
        im_info = blobs['im_info']
        gt_boxes = blobs['gt_boxes']
        gt_ishard = blobs['gt_ishard']
        dontcare_areas = blobs['dontcare_areas']

        # Get cross-entropy and box loss for rpn and faster-rcnn networks
        # Since the RPN is in training mode, this will create rois that are 
        # partly from GT and partly from RPN.
        cls_prob, bbox_pred, rois, features, rpn_ce, rpn_lb, f_ce, f_lb, \
                roi_data = \
                faster_rcnn_forward(
                        self.model.detection_branch, im_data, im_info, gt_boxes,
                        gt_ishard, dontcare_areas)
        ret.update({
                "rpn_ce": rpn_ce,
                "rpn_lb": rpn_lb,
                "f_ce": f_ce,
                "f_lb": f_lb})

        # TODO normally, we will get ROIs from elsewhere. When that happens,
        # move this code.
        # Desire: image, gt boxes w/ class labels, roi boxes with max overlap
        # classes.
        #import pdb; pdb.set_trace()
        """
        self.visualizer.visualize_samples(
                im_data, 
                roidb[0]["gt_classes"],
                roidb[0]["gt_overlaps"],
                gt_boxes[:, 0:4])
        """

        # Find human boxes that have >= 0.5 overlap with gt
        # RB has elements; want rb[0]['gt_boxes']
        # TODO these person_indexes are empty (?)

        """
        # TODO sad old confused code.
        person_index = self.vcoco_translator.nouns_2_ids["person"]
        elem = roidb[0]
        candidate_persons = np.where(np.logical_and(
                elem["gt_classes"] == person_index,
                elem["gt_overlaps"][:, person_index] > 0.5))
        # TODO b_h is empty...
        # TODO this data that we're feeding is WAY wrong. the filename
        # corresponds to a picture of a surfer, labels show airplanes...
        b_h = elem["boxes"][candidate_persons]
        try:
            # TODO is this causing an error?
            np.random.shuffle(b_h)
        except Exception as e:
            import pdb; pdb.set_trace()
        b_h = b_h[:16] # only choose 16 boxes.
        b_h = np.array([[1., 1., 2., 2.]]) # TODO :/ the candidate person boxes
        # are not found. So I'm hallucinating these values for now :(
        b_h = network.np_to_variable(b_h)
        """

        # roi_data consists of:
        # rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
        person_index = self.vcoco_translator.nouns_2_ids["person"]
        person_roi_indices = roi_data[1] == person_index
        # Choose at most 16 people
        nonz = torch.nonzero(person_roi_indices.data.squeeze().cpu()) \
                .squeeze().numpy()
        if nonz.size != 0:
            choices = np.random.choice(nonz, min(nonz.size, 16), replace=False)
            person_roi_indices = torch.from_numpy(choices).cuda(0)
            #person_roi_indices = person_roi_indices[choices]
            b_h = rois[person_roi_indices]

            action_scores, action_locations = self.model.human_centric_branch(
                    b_h, features)

            # Get ground-truth and calculate loss.
            # This is [B=1 x NActions]
            gt_action_scores = self.vcoco_translator.get_action_labels(vcoco_ann)
            gt_action_scores = network.np_to_variable(gt_action_scores)
            gt_action_scores = gt_action_scores.unsqueeze(0).expand_as(
                    action_scores)

            action_ce = F.binary_cross_entropy(action_scores, gt_action_scores)

            ret.update({
                    "action_ce": action_ce,
                })

        # Get ground-truth role locations for non-agent roles.
        # It will be a [B=1 x NActionNonagentRoles x 5] structure]
        # The actions for 2-obj action things are treated uniquely.
        # TODO we probably want to make the GT labels relative to the agent?
        gt_action_locations = \
                self.vcoco_translator.get_action_nonagent_role_locations(
                        vcoco_ann)
        gt_action_locations = gt_action_locations.squeeze(0)
        # (it's a np.ndarray with size [1 x NActionRolesNonagent x 4])
        # Choose the action locations that correspond to a ground-truth action
        chosen_locations = np.where(np.logical_and(
                gt_action_locations[:, 0] == 1,
                np.logical_not(np.isnan(gt_action_locations[:, 1]))))
        assert len(chosen_locations) == 1, "Expected size-1 tuple"
        gt_action_locations = gt_action_locations[chosen_locations[0], 1:]

        if gt_action_locations.size != 0:
            try:
                print "SIZE IS: ", gt_action_locations.size
                gt_action_locations = network.np_to_variable(
                        gt_action_locations).unsqueeze(0)

                action_locations = action_locations.cpu().data.numpy()
                action_locations = action_locations[:, chosen_locations[0], :]
                action_locations = network.np_to_variable(action_locations)

                # Expand in the batch dimension.
                gt_action_locations = gt_action_locations.expand_as(action_locations)

                # It's possible that there are no actions with localized information.
                if gt_action_locations.dim() != 0:
                    location_l1 = F.smooth_l1_loss(
                            action_locations, gt_action_locations)

                ret.update({
                        "location_l1": location_l1,
                    })
            except:
                import pdb; pdb.set_trace()

        # TODO continue here with the editing / debugging.

        # TODO the last part is confusing. I'll take it to mean that b_h and b_o
        # must both be taken from ground truth labels.
        # (But possibly, they mean that only the cases where the action has a
        # positive label for those boxes)
        # Get the gT human box.
        # TODO consider removing this part of the system...?
        # TODO this will give another gradient to human branch... ???
        # gt_actions will probably be one-hot along each row for those
        # interactions? Or not...; can just expand gt_action_scores from above.
        b_h, b_o, gt_actions = self.vcoco_translator.get_human_object_gt_pairs(
                vcoco_ann)
        if b_h is not None:
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
        #vcoco_all = vu.load_vcoco("vcoco_train")
        #categories = [x["name"] for x in vu.load_coco().cats.itervalues()]
        translator = ld.VCocoTranslator(
                dataloader.vcoco_all, dataloader.get_classes())
        self.loss = HoiLoss(model, translator, self.epoch_data)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.SGD(parameters, lr=self.learn_rate)
        self.epoch_callbacks = [md.save_trainer, epoch_log_error]
        self.batch_callbacks = [iter_log_error, print_batch_loss]

    def handle_batch(self, data):
        # TODO figure out how to best parallelize this
        x, y = data
        """
        x = ag.Variable(x)
        y = ag.Variable(y)
        if self.cuda:
            x = x.cuda(self.cuda[0])
            y = y.cuda(self.cuda[0])
        """
        self.optimizer.zero_grad()
        loss = self.loss([x], y)
        loss.backward()
        self.optimizer.step()

class Visualizer(object):
    def __init__(self, vcoco_translator):
        self.vcoco_translator = vcoco_translator

    def vis_sample(self, roi, im_data, cls_label):
        assert roi.size == 5, "expected roi with size 5"
        roi = roi[1:]
        #im_data = im_data[0, :, :, :]
        #im = im_data.copy()
        #im = im.transpose(1, 2, 0).copy()
        #im += cf.cfg.PIXEL_MEANS
        #im = im[:, :, (2, 1, 0)]
        #im = im.astype(np.uint8)
        print im_data.shape
        print cfg.PIXEL_MEANS.shape
        #im = im_data[0, :, :, :].transpose((1, 2, 0)).copy()


        im = im_data[0, :, :, :]#.transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)

        #im += 0
        #im += cfg.PIXEL_MEANS
        #im = im[:, :, :, (0, 3, 2, 1)]
        cls_name = self.vcoco_translator.ids_2_nouns[cls_label]
        print "class name", cls_name

        plt.imshow(im)
        plt.gca().add_patch(
            plt.Rectangle(
                    (roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                    fill=False, edgecolor='r', linewidth=3))
        plt.show()

    def visualize_samples(self, im_data, gt_classes, gt_overlaps, rois):
        #import pdb; pdb.set_trace()
        print im_data.shape
        print cfg.PIXEL_MEANS.shape
        print rois.shape
        assert rois.shape[1] == 4, \
                "Expected rois.shape[1]==4, rois.shape is " % str(rois.shape)

        im = im_data[0, :, :, :]
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        plt.imshow(im)

        for i, bigroi in enumerate(rois):
            #roi = bigroi[1:]
            roi = bigroi
            plt.gca().add_patch(
                plt.Rectangle(
                        (roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                        fill=False, edgecolor='r', linewidth=3))
            gt_class = self.vcoco_translator.ids_2_nouns[gt_classes[i]]
            gt_overlap = gt_overlaps[i, gt_classes[i]]
            label = "%s | %.3f" % (gt_class, gt_overlap)
            plt.gca().text(roi[0], roi[1], label, backgroundcolor='w', size=8)
        plt.show()
