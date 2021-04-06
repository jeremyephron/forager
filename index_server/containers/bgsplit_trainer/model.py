import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision

class Model(nn.Module):
    def __init__(self, num_main_classes, num_aux_classes):
        super(Model, self).__init__()
        self.backbone = vision.models.resnet50(pretrained=True)
        backbone_feature_dim = 0
        self.main_head = nn.Linear(backbone_feature_dim, num_main_classes)
        self.auxiliary_head = nn.Linear(backbone_feature_dim, num_aux_classes)

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):
        feature = self.forward_backbone(x)
        main_logits = self.main_head(feature)
        aux_logits = self.auxiliary_head(feature)
        return main_logits, aux_logits

    def predict(self, x):
        feature = self.forward_backbone(x)
        main_logits = self.main_head(feature)
        return F.softmax(main_logits)
