import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, emb_dim=128):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet50.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.flattener = nn.Flatten()

    def forward(self, x):
        h = self.features(x)
        h = self.flattener(h)
        return h

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        z = self.projector(x)
        return z

class SimCLR_Model(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SimCLR_Model, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
