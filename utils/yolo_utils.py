import torch
import torch.nn as nn
# from torchmetrics import JaccardIndex as IoU

def IoU(box1, box2):
    # Calculate the coordinates of the top-left and bottom-right corners for each box
    xmin1 = box1[..., 0] - box1[..., 2] / 2
    ymin1 = box1[..., 1] - box1[..., 3] / 2
    xmax1 = box1[..., 0] + box1[..., 2] / 2
    ymax1 = box1[..., 1] + box1[..., 3] / 2

    xmin2 = box2[..., 0] - box2[..., 2] / 2
    ymin2 = box2[..., 1] - box2[..., 3] / 2
    xmax2 = box2[..., 0] + box2[..., 2] / 2
    ymax2 = box2[..., 1] + box2[..., 3] / 2

    # Calculate the intersection area
    intersection_width = torch.clamp_min(torch.min(xmax1, xmax2) - torch.max(xmin1, xmin2), min=0)
    intersection_height = torch.clamp_min(torch.min(ymax1, ymax2) - torch.max(ymin1, ymin2), min=0)
    intersection_area = intersection_width * intersection_height

    # Calculate the union area
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        target = target.reshape(-1, self.S, self.S, self.C + 5)

        iou_b1 = IoU(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = IoU(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # print(iou_b1.shape, iou_b2.shape, ious.shape, bestbox.shape)
        exists_box = target[..., 20].unsqueeze(-1)

        # For Box Coordinates
        box_predictions = exists_box * (
            (
                bestbox.unsqueeze(-1) * predictions[..., 26:30]
                + (1 - bestbox.unsqueeze(-1)) * predictions[..., 21:25]
            )
        )
        
        box_targets = exists_box * target[..., 21:25]
        
        box_predictions[..., 2:4] = (
            torch.sign(box_predictions[..., 2:4]) *
            torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        # For Object Loss
        pred_box = (bestbox.unsqueeze(-1) * predictions[..., 25:26] + (1 - bestbox.unsqueeze(-1)) * predictions[..., 20:21])
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        # For No Object Loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        # For Class Loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

class YoloSceduler:
    def __init__(self, optimizer, warmup=5):
        self.warmup = warmup
        self.reset(optimizer)

    def change(self):
        red = 1
        if self.step < self.warmup:
            new_lr = 1e-3 + self.step * (1e-2 - 1e-3) / self.warmup
            self.optimizer.param_groups[0]["lr"] = new_lr * red
        elif self.step == self.warmup:
            self.optimizer.param_groups[0]["lr"] = 1e-2 * red
        elif self.step == self.warmup + 75:
            self.optimizer.param_groups[0]["lr"] = 1e-3 * red
        elif self.step == 75 + 30 + self.warmup:
            self.optimizer.param_groups[0]["lr"] = 1e-4 * red
        self.step += 1
    
    def reset(self, optimiser):
        self.optimizer = optimiser
        self.step = 0