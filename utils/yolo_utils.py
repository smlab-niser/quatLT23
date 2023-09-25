import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm

class YoloSceduler:
    def __init__(self, optimizer, reduction, dropat = [75, 75+30], dropby=0.1, warmup=5):
        self.warmup = warmup
        self.red = reduction * 1e2  # bacause lr is multiplied by 1e-2 in the hparams dictinary
        self.dropat = dropat
        self.dropby = dropby
        self.reset(optimizer)

    def change(self):
        if self.step <= self.warmup:
            new_lr = 1e-3 + self.step * (1e-2 - 1e-3) / self.warmup
            self.optimizer.param_groups[0]["lr"] = new_lr * self.red
            print(f"Changing lr to {self.optimizer.param_groups[0]['lr']} at step {self.step}")
        elif (self.step-self.warmup) in self.dropat:
            self.optimizer.param_groups[0]["lr"] *= self.dropby
            print(f"Changing lr to {self.optimizer.param_groups[0]['lr']} at step {self.step}")
        self.step += 1

    def reset(self, optimiser):
        self.optimizer = optimiser
        self.step = 0




# ================================================== #
#         Next part of the code is modified          #
#       from Aladdin Persson's code on YOLOv1        #
# GitHub: aladdinpersson/Machine-Learning-Collection #
# ================================================== #


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
        self.mse = nn.MSELoss(reduction="sum")
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
        



# ====== Rest part used for evaluation ====== #

def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IoU(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
            )
            

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes_30(predictions, S=7):
    """Converts bounding box predictions into cell indices and offsets.

    Args:
        predictions (array): predictions from the yolo model of shape (batch_size, 7, 7, 30)
        S (int, optional): grid size. Defaults to 7.

    Returns:
        array: cell indices and offsets of shape (batch_size, 7, 7, 6)
                here 6 = [class, confidence, x, y, w, h]
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box_index = scores.argmax(0).unsqueeze(-1)
    best_box = bboxes1 * (1 - best_box_index) + bboxes2 * best_box_index
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_box[..., 0:1] + cell_indices)
    y = 1 / S * (best_box[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_box[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    return torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)


def convert_cellboxes_25(predictions, S=7):
    """Converts bounding box predictions into cell indices and offsets.

    Args:
        predictions (array): predictions from the yolo model of shape (batch_size, 7, 7, 25)
        S (int, optional): grid size. Defaults to 7.

    Returns:
        array: cell indices and offsets of shape (batch_size, 7, 7, 6)
                here 6 = [class, confidence, x, y, w, h]
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 25)
    best_box = predictions[..., 21:25]
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_box[..., 0:1] + cell_indices)
    y = 1 / S * (best_box[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_box[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = predictions[..., 20].unsqueeze(-1)
    return torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)


def cellboxes_to_boxes(out, S=7):
    convert_fn = convert_cellboxes_30
    if out.shape[-1] == 25:
        convert_fn = convert_cellboxes_25
    converted_pred = convert_fn(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    return converted_pred.tolist()



# writen by me

def mAP(model, data_loader, device="cuda", iou_threshold_nms=0.5, threshold_nms=0.5, iou_threshold_map=0.5):
    model.eval()
    with torch.no_grad():
        pred_boxes, target_boxes = get_bboxes(data_loader, model, iou_threshold_nms, threshold_nms, device)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold_map)
    model.train()
    return mean_avg_prec.item()*100

