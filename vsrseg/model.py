import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchvision as tv
import tensorflow as tf

import utils.mylogger as logging
import utils.methods as mt

IMSIZE = (224, 224) # width, height

class BBTrainer(object):
    """
    Train a model to predict bounding boxes and semantic information from 
    images.
    """
    def __init__(self, model, dataloader, cuda=True, lr=0.001, log_dir=None):
        self.epoch = 0
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr)
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.log_dir + "/train")

    def train(self, epochs):
        # TODO need to come up with loss function, optimization method, etc. 
        # that we want to use.
        # TODO maybe the loss function should be part of the model, and this
        # training loop will just leverage it and output graphs / etc.
        logging.getLogger(__name__).info(
                "Starting training at epoch %d" % self.epoch)
        for self.epoch in range(self.epoch, self.epoch + epochs):
            logging.getLogger(__name__).info("Running epoch %d" % self.epoch)
            history = []
            # Make a full pass over the training set.
            for i, data in enumerate(self.dataloader):
                history.append(self.handle_batch(data))
            if mt.should_do(self.epoch, 1) and (self.log_dir is not None):
                err = sum(history) / len(history)
                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="error", simple_value=err),
                        tf.Summary.Value(tag="epoch", simple_value=self.epoch)])
                self.train_writer.add_summary(summary, self.epoch)


            logging.getLogger(__name__).info(
                    "---> Epoch %d: Loss=%.8f" % (
                            self.epoch, sum(history)/len(history)))

    def handle_batch(self, data):
        x, y = data
        x = ag.Variable(x)
        y = ag.Variable(y)
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
        pred = self.model(x)
        err = self.criterion(pred, y)
        err.backward()
        self.optimizer.step()
        return err.data[0]

class CtxBB(nn.Module):
    # TODO this model will be changed to predict bounding boxes for objects.
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
