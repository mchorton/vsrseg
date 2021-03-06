"""
Skeleton code for building and training models. The models in here are mostly
toys for unit/integration testing purposes, or classes to be subclassed.
"""

import os
import itertools as it
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchvision as tv
import torchvision.transforms as tt
import tensorflow as tf

import utils.mylogger as logging
import utils.methods as mt

IMSIZE = (224, 224) # width, height

# Idea: add pre- and post-loop hooks for logging, printing info, etc.
class Trainer(object):
    """
    Train a model to predict bounding boxes and semantic information from 
    images.

    To make a new Trainer, subclass this model and consider changing:
    - RelevantKwargs
    - self.criterion
    - self.optimizer
    - self.callbacks; this is a list of callbacks that take a Trainer as their
        sole argument. This is for doing things like saving models, printing
        losses, etc.
    """
    RelevantKwargs = {
            "cuda": [],
            "save_dir": None,
            "save_per": 1,
            "learn_rate": 0.001,
        }
    def __init__(self, model, dataloader, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.model = model
        self.dataloader = dataloader
        my_kwargs = copy.deepcopy(self.RelevantKwargs)
        for kw, setting in my_kwargs.iteritems():
            if kw in kwargs:
                setting = kwargs[kw]
            setattr(self, kw, setting)

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Used to store information pertaining to the current epoch.
        self.epoch_data = []
        # Used to store epoch_data through multiple epochs.
        self.history = []
        self.epoch_callbacks = []
        self.batch_callbacks = []

        if self.cuda:
            self.model = self.model.cuda(self.cuda[0])

    def run_batch_callbacks(self):
        for callback in self.batch_callbacks:
            callback(self)

    def run_epoch_callbacks(self):
        for callback in self.epoch_callbacks:
            callback(self)

    def fractional_epoch(self):
        return (self.epoch - 1) + (1. * self.iteration / len(self.dataloader))

    def total_iterations(self):
        print "Calling total_iterations"
        return (self.epoch - 1) * len(self.dataloader) + self.iteration

    def train(self, epochs):
        self.model.train()
        logging.getLogger(__name__).info(
                "Starting training at epoch %d" % (self.epoch + 1))
        for self.epoch in range(self.epoch + 1, self.epoch + 1 + epochs):
            logging.getLogger(__name__).info("Running epoch %d" % self.epoch)
            # Make a full pass over the training set.
            for self.iteration, data in enumerate(self.dataloader, start=1):
                self.handle_batch(data)
                self.run_batch_callbacks()
            self.run_epoch_callbacks()
            del self.epoch_data[:] # Clear the list of epoch data

    def handle_batch(self, data):
        raise NotImplementedError

class BasicTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)
        self.criterion = nn.MSELoss()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.learn_rate)
        self.epoch_callbacks = [print_loss, log_error, save_trainer]

        if self.save_dir is not None:
            self.train_writer = tf.summary.FileWriter(
                    os.path.join(self.save_dir, "train/"))

    def handle_batch(self, data):
        x, y = data
        x = ag.Variable(x)
        y = ag.Variable(y)
        if self.cuda:
            x = x.cuda(self.cuda[0])
            y = y.cuda(self.cuda[0])
        pred = self.model(x)
        err = self.criterion(pred, y)
        self.optimizer.zero_grad()
        err.backward()
        self.optimizer.step()
        return err.data[0]

class TestTrainerRealData(Trainer):
    def __init__(self, *args, **kwargs):
        super(TestTrainerRealData, self).__init__(*args, **kwargs)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learn_rate)
        self.epoch_callbacks = [print_loss, log_error, save_trainer]

    def handle_batch(self, data):
        x, y = data
        x = ag.Variable(x)
        y = y[1]["verbs"]["throw"]["label"].float()
        y = ag.Variable(y)
        if self.cuda:
            x = x.cuda(self.cuda[0])
            y = y.cuda(self.cuda[0])
        pred = self.model(x)
        err = self.criterion(pred, y)
        err.backward()
        self.optimizer.step()

def log_error(trainer):
    # log error values
    if len(trainer.epoch_data) < 1:
        return
    if mt.should_do(trainer.iteration, 1) and (trainer.save_dir is not None):
        err = trainer.epoch_data[-1]
        values = [
                tf.Summary.Value(tag="error", simple_value=err),
            ]
        values += trainer.model.get_summary_values()
        summary = tf.Summary(value=values)
        trainer.train_writer.add_summary(summary, trainer.fractional_epoch())

def save_trainer(trainer):
    # save the model if needed
    if mt.should_do(trainer.epoch, trainer.save_per) and \
            trainer.save_dir is not None:
        logging.getLogger(__name__).info("Saving training file...")
        torch.save(trainer, os.path.join(
                trainer.save_dir, "trainer_%d.trn" % trainer.epoch))

def print_loss(trainer):
    logging.getLogger(__name__).info(
            "---> After Epoch #%d: Loss=%.8f" % (
                    trainer.epoch,
                    sum(trainer.local_history) / len(trainer.local_history)))

class CtxBB(nn.Module):
    def __init__(self):
        super(CtxBB, self).__init__()
        self.resnet = tv.models.resnet18(pretrained=True)
        self.layer = nn.Linear(1000, 1)

    def get_features(self, image):
        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)

        return x

    def forward(self, image):
        ret = self.get_features(image)
        ret = ret.view(ret.size(0), -1)
        ret = self.resnet.fc(ret)
        return F.sigmoid(self.layer(ret))

    def get_summary_values(self):
        # Visualize the first convolutional filter
        # this is a 3d tensor (depth x width x height)
        filter_tensor = self.resnet.conv1.weight[0]
        width, height = filter_tensor.size()[1:]
        pilimage = tt.ToPILImage()(filter_tensor.data.cpu())
        img_bytes = mt.pil_2_bytes(pilimage)

        # TODO this output isn't useful.
        img = tf.Summary.Image(width=width, height=height, colorspace=3,
                encoded_image_string=img_bytes) 
        img_value = tf.Summary.Value(tag="First Filter", image=img)
        ret = [img_value]

        # Get the magnitude of updates of weights in the last layer.
        if hasattr(self, "prev_weights"):
            cur_diff = self.layer.weight - self.prev_weights
            update_mag = torch.norm(cur_diff)
            ret.append(tf.Summary.Value(
                    tag="update_mag", simple_value=update_mag.data[0]))

            # calculate the agreement between the last vector and this vector.
            agreement = mt.cos_angle_var(self.layer.weight, self.prev_weights)
            ret.append(tf.Summary.Value(
                    tag="cos(theta) of weights", simple_value=agreement))

            if hasattr(self, "prev_diff"):
                # calculate the agreement between the last and cur. updates
                agreement = mt.cos_angle_var(cur_diff, self.prev_diff)
                ret.append(tf.Summary.Value(
                    tag="cos(theta) of updates", simple_value=agreement))
            self.prev_diff = cur_diff
        self.prev_weights = self.layer.weight.clone()

        ret.append(tf.Summary.Value(
                tag="sanity_0s", simple_value=mt.cos_angle(torch.ones(4),
                torch.zeros(4))))
        ret.append(tf.Summary.Value(
                tag="sanity_neg1", simple_value=mt.cos_angle(torch.ones(4),
                torch.ones(4) * -1)))
        return ret

    def interpret_predictions(self, predictions, labels):
        # TODO this is just a skeleton for now.
        # Turn a tensor of predictions into a list of predictions.
        # predictions: a list of array values to be turned into predictions
        # labels: a list of the labels for this image.
        #     labes[0]: a the coco annotations
        #     labels[1]: the vcoco annotations; e.g.,
        #         {"image_id": int, "verbs": {verb_name: stuff}}
        ret = []
        for i, (prediction) in enumerate(predictions):
            p = {
                    "image_id": int(labels[1]["image_id"][i]),
                    "person_box": [0, 0, 10, 10],
                }
            for verb_name in labels[1]["verbs"]:
                p["%s_agent" % verb_name] = 0.1
                p["%s_obj" % verb_name] = 0.1
                p["%s_instr" % verb_name] = 0.1
            ret.append(p)
        return ret

class TestCtxBB(CtxBB):
    def __init__(self, *args, **kwargs):
        super(TestCtxBB, self).__init__(*args, **kwargs)
        stride = 16
        self.feature_map = nn.Conv2d(3, 1, 1, stride=stride)
        self.linear = nn.Linear(IMSIZE[0] * IMSIZE[1] / stride / stride, 1)

    def forward(self, image): 
        ret = self.feature_map(image)
        ret = ret.view(ret.size(0), -1)
        return F.sigmoid(self.linear(ret))
