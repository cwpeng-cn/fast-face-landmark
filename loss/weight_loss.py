import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, landmark_gt, landmarks, visables):
        # landmark_gt => N,64,2
        # visables => N,64
        l2_distant = torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=-1)
        l2_distant = l2_distant*visables
        return torch.mean(l2_distant)
    
if __name__=="__main__":
    loss=WeightLoss()
    landmark_gt=torch.rand(2,64,2)
    landmarks=torch.rand(2,64,2)
    visables=torch.ones(2,64)
    value=loss(landmark_gt,landmarks,visables)
    print(value)