import torch
import torch.nn as nn
import torch.autograd as ag

import utils.mylogger as logging

class BBTrainer(object):
    """
    Train a model to predict bounding boxes and semantic information from 
    images.
    """
    def __init__(self, model, dataloader, cuda=True, lr=0.001):
        self.epoch = 0
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr)
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

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
        self.layer = nn.Linear(128 * 128 * 3, 1)

    def forward(self, image):
        ret = self.layer(image.view(-1, 128 * 128 * 3))
        return ret
