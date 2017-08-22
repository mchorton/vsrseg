from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision as tv
from faster_rcnn.faster_rcnn import FasterRCNN
import faster_rcnn.network as network

"""
def get_resnet50_frcnn():
    # Create a FasterRCNN with ResNet50 backbone
    model = FasterRCNN()
    model.rpn.features = tv.
"""

def load_pretrained_r50frcnn(r50frcnn):
    # params is a state_dict.
    params = model_zoo.load_url(tv.models.resnet.model_urls['resnet50'])
    #r50_dict = r50frcnn.rpn.features.state_dict()

    r50frcnn.rpn.features.load_state_dict(params)
    """
    for name, val in r50_dict.iteritems():
        # TODO don't think I need to skip 'bn.'
        if name.find('bn') >= 0:
            continue
        param = r50_dict[name]
        ptype = 'weights' if name[-1] == 't' else 'biases'
        if ptype == 'weights':
            try:
                param = param.permute(3, 2, 0, 1)
            except Exception as e:
                import pdb; pdb.set_trace()
        val.copy_(param)
    """
    #frcnn_dict = r50frcnn.state_dict()
    #frcnn_dict['fc.weight'].copy_(params['fc.weight'])
    #frcnn_dict['fc.bias'].copy_(params['fc.bias'])

    #r50frcnn.load_state_dict(
    #        {k: params[k] for k in ["fc.weight", "fc.bias"]})

    # Copy FC weights from last layer of r50, to the FC layer
    """
    pairs = {'fc.fc': 'fc'}
    for k, v in pairs.iteritems():
        key = '{}.weight'.format(k)
        param = params[v]['weights'].permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = params[v]['biases']
        frcnn_dict[key].copy_(param)
    """

class R50FasterRCNN(FasterRCNN):
    def __init__(self, pretrained_r50=False, *args, **kwargs):
        super(R50FasterRCNN, self).__init__(*args, **kwargs)
        # Instead of an fc6 and fc7 layer, this RCNN has a single FC layer
        delattr(self, "fc6")
        delattr(self, "fc7")
        self.rpn.features = resnet50_conv()
        self.fc = nn.Linear(
                512 * tv.models.resnet.BasicBlock.expansion, 4096)

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        """
        Similar to Faster RCNN forward function, but without fc6/fc7, and with a
        single fc to replace them.
        """
        #import pdb; pdb.set_trace()
        features, rois = self.rpn(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            rois = roi_data[0]

        # roi pool
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = F.relu(self.fc(x))
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

class R50Conv(tv.models.ResNet):
    """
    Like ResNet50, but the forward() function doesn't have an FC layer.
    """
    def __init__(self, *args, **kwargs):
        super(R50Conv, self).__init__(*args, **kwargs)
        self.rescale = tv.transforms.Scale((224, 224))

    def forward(self, x):
        import pdb; pdb.set_trace()
        # TODO: ok, it almost works. Just need to:
        # 1. Figure out how/why VGG is OK with this image size.
        # 2. Figure out how to do the same thing in ResNet

        #x = x.permute(0, 2, 3, 1)
        
        #x = self.rescale(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 

        return x

def resnet50_conv(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = R50Conv(tv.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
                tv.models.resnet.model_urls['resnet50']))
    return model
