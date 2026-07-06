import torch
import torch.nn as nn
import networks.all_models as all_models


class FedAvgResNet18(nn.Module):
    """
    ResNet-18 classifier used for local_train_mode == 'fedavg' (classic FedAvg-style backbone).
    Forward API matches efficientb0: (logits, logits) or (projector_feats, logits) when project=True.
    """

    def __init__(self, n_classes, args=None):
        super().__init__()
        self.n_classes = n_classes
        self.args = args
        self.model = all_models.get_model("Resnet18", pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.n_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs),
            nn.Linear(self.num_ftrs, 1024),
        )

    def _penultimate(self, inputs):
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, inputs, project=False):
        feat = self._penultimate(inputs)
        if not project:
            logits = self.model.fc(feat)
            return logits, logits
        features = self.projector(feat)
        logits = self.model.fc(feat)
        return features, logits


# FedProx 与 FedAvg 共用同一 ResNet-18 结构；算法差异在局部目标 (mu) 与聚合入口名，不在骨干网络。
FedProxResNet18 = FedAvgResNet18

# FedFSA 参考实现同样使用 ResNet-18 / CNN 分类头；FSA 与动量在 ``train_FedFSA`` / ``fedfsa_aggregate``。
FedFSAResNet18 = FedAvgResNet18


class efficientb0(nn.Module):
    def __init__(self, n_classes, args=None):
        super(efficientb0, self).__init__() 
        self.n_classes = n_classes
        self.args = args
        self.model = all_models.get_model("Efficient_b0", pretrained=True)
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, self.n_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs),
            nn.Linear(self.num_ftrs, 1024)
        )

    def forward(self, inputs, project=False):
        if project == False:
            # Convolution layers
            x = self.model.extract_features(inputs)
            # Pooling and final linear layer
            x = self.model._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.model._dropout(x)
            x = self.model._fc(x)
            return x, x
        else:
            # Convolution layers
            x = self.model.extract_features(inputs)
            # Pooling and final linear layer
            x = self.model._avg_pooling(x)
            x = x.flatten(start_dim=1)
            features = self.projector(x)
            y = self.model._dropout(x)
            y = self.model._fc(y)
            return features, y
