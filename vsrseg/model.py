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
                values = [
                        tf.Summary.Value(tag="error", simple_value=err),
                    ]
                values += self.model.get_summary_values()
                summary = tf.Summary(value=values)
                self.train_writer.add_summary(summary, self.epoch + 1)

            logging.getLogger(__name__).info(
                    "---> After Epoch #%d: Loss=%.8f" % (
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

    def get_summary_values(self):
        # Visualize the first convolutional filter
        # this is a 3d tensor (depth x width x height)
        filter_tensor = self.resnet.conv1.weight[0]
        width, height = filter_tensor.size()[1:]
        pilimage = tt.ToPILImage()(filter_tensor.data.cpu())
        img_bytes = mt.pil_2_bytes(pilimage)

        img = tf.Summary.Image(width=width, height=height, colorspace=3,
                encoded_image_string=img_bytes) # TODO wrong.
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
