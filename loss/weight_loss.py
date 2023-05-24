import torch
from torch import nn
from torch.nn import BCELoss


class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()
        self.class_criterion = BCELoss()

    def forward(self, landmark_gt, landmarks, visables, pre_visable,
                actually_visable):
        """
            l2 loss + bce loss
        Args
            @landmark_gt: gt landmark
            @landmarks: predict landmark
            @visable: landmark gt visibility (original image)
            @pre_visable: predict landmark visibility
            @actually_visable: landmark gt visibility (croped,scaled image)
        """

        l2_distant = torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=-1)
        l2_distant = l2_distant * visables
        class_loss = self.class_criterion(pre_visable, actually_visable)
        return torch.mean(l2_distant) + 0.05 * class_loss


if __name__ == "__main__":
    loss = WeightLoss()
    landmark_gt = torch.rand(2, 64, 2)
    landmarks = torch.rand(2, 64, 2)
    visables = torch.ones(2, 64)
    value = loss(landmark_gt, landmarks, visables, visables * 1.0, visables)
    print(value)