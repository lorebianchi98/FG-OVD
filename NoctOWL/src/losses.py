import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np

from copy import deepcopy
from src.matcher import HungarianMatcher, box_iou, generalized_box_iou

# Code borrowed and adapted from https://github.com/lorebianchi98/FG-CLIP/blob/main/src/loss.py
class Contrastive(nn.Module):
    def __init__(self, margin=0.1, max_violation=False, ltype='triplet'):
        super(Contrastive, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.ltype = ltype
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, scores):
        if self.ltype == 'infonce':
            # Scaling logits
            scale = self.logit_scale.exp()
            logits_per_image = scale * scores

            # Labels for cross-entropy: point all rows to the first column (index 0)
            labels_image = torch.zeros(logits_per_image.shape[0], device=logits_per_image.device, dtype=torch.long)

            # Compute unidirectional CE loss
            loss = F.cross_entropy(logits_per_image, labels_image) 
                        
        elif self.ltype == 'triplet':
            # in addition we perform only triplet loss keeping the image as anchor (row-wise optimization)
            positive_scores = scores[:, 0].view(scores.size(0), 1)
            d1 = positive_scores.expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)

            # mask with all elements True in the first column
            mask = torch.cat((torch.ones(scores.shape[0], 1), torch.zeros(scores.shape[0], scores.shape[1] - 1)), dim=1) > 0.5
            I = mask
            if torch.cuda.is_available():
                I = I.to(scores.device)
            cost_s = cost_s.masked_fill_(I, 0)

            # keep the maximum violating negative for each query
            if self.max_violation:
                cost_s = cost_s.max(1)[0]

            loss = cost_s.mean()
            
        return loss

class PushPullLoss(torch.nn.Module):
    def __init__(self, n_classes, margin=0.2, scales=None, class_ltype='triplet', self_distillation_loss=None):
        super().__init__()
        self.matcher = HungarianMatcher(n_classes)
        self.class_ltype = class_ltype
        self.class_criterion = torch.nn.BCELoss(reduction="none", weight=scales)
        self.contrastive = Contrastive(margin=margin, ltype=class_ltype)
        self.background_label = n_classes
        if self_distillation_loss == 'mse':
            self.self_distillation_loss = nn.MSELoss()
        elif self_distillation_loss == 'ce':
            self.self_distillation_loss = nn.CrossEntropyLoss()
        else:
            self.self_distillation_loss = None

    def class_loss(self, outputs, target_classes):
        """
        Custom loss that works off of similarities
        """
        src_logits = torch.abs(outputs["pred_logits"])
        src_logits = src_logits.transpose(0, 2).transpose(1, 2)

        pred_logits = src_logits[:, target_classes != self.background_label].t()
        bg_logits = src_logits[:, target_classes == self.background_label].t()
        target_classes = target_classes[target_classes != self.background_label]
        # Positive loss
        if self.class_ltype == 'BCE':
            pos_targets = torch.nn.functional.one_hot(target_classes, self.background_label)
            pos_loss = self.class_criterion(pred_logits, pos_targets.float())
            pos_loss = (torch.pow(1 - torch.exp(-pos_loss), 2) * pos_loss).sum(dim=1).mean()
        
        elif self.class_ltype == 'triplet' or self.class_ltype == 'infonce':
            pos_loss = self.contrastive(pred_logits)
        # Negative loss
        neg_targets = torch.zeros(bg_logits.shape).to(bg_logits.device)
        neg_loss = self.class_criterion(bg_logits, neg_targets)
        neg_loss = (torch.pow(1 - torch.exp(-neg_loss), 2) * neg_loss).sum(dim=1).mean()

        return pos_loss, neg_loss

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes):
        """
        (DETR box loss)

        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = torch.nn.functional.l1_loss(
            src_boxes, target_boxes, reduction="none"
        )

        metadata = {}

        loss_bbox = loss_bbox.sum() / num_boxes
        metadata["loss_bbox"] = loss_bbox.tolist()

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def listnet_loss(self, pred_scores, target_scores):
        ''' 
        The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
        
        Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
        Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
        '''
        return torch.mean(-torch.mean(F.softmax(target_scores, dim=1) * F.log_softmax(pred_scores, dim=1), dim=1))
        

    def forward(
        self,
        predicted_classes,
        target_classes,
        predicted_boxes,
        target_boxes,
        lvis_pred_scores=None,
        lvis_target_scores=None
    ):
        assert (lvis_pred_scores is None and lvis_target_scores is None) or (lvis_pred_scores is not None and lvis_target_scores is not None), "Target or predicted LVIS scores missing!"
        
        bs, n_preds = predicted_boxes.shape[:2]
        # Format to detr style
        in_preds = {
            "pred_logits": predicted_classes,
            "pred_boxes": predicted_boxes,
        }

        in_targets = [
            {"labels": _labels, "boxes": _boxes}
            for _boxes, _labels in zip(target_boxes, target_classes)
        ]

        target_classes, indices, idx = self.matcher(in_preds, in_targets)

        loss_bbox, loss_giou = self.loss_boxes(
            in_preds,
            in_targets,
            indices,
            idx,
            num_boxes=sum(len(t["labels"]) for t in in_targets),
        )
        
        # putting the target class to all the bboxes which intersecate with the matching boxes
        matched_boxes = predicted_boxes[target_classes != self.background_label] # BS x 4
        iou = ops.box_iou(matched_boxes, predicted_boxes.view(-1, 4)).view(bs, bs, n_preds) # BS x BS x 2304
        iou = iou[torch.eye(bs) > 0.5] # BS x 2304, keeping only the iou coming from the same image
        idx = iou > 0.85
        target_classes[idx] = 0
        
        loss_class, loss_background = self.class_loss(in_preds, target_classes)

        # we calculate the rank loss
        if lvis_pred_scores is not None:
            lvis_target_scores = lvis_target_scores[target_classes != self.background_label]
            lvis_pred_scores = lvis_pred_scores[target_classes != self.background_label]
            # we want to calculate the ListNet loss only on the subset of classes with the highest score on at least one row
            # to_keep_classes = torch.unique(torch.argmax(lvis_target_scores, dim=1))
            # lvis_pred_scores = lvis_pred_scores[:, to_keep_classes]
            # lvis_target_scores = lvis_target_scores[:, to_keep_classes]
            # loss_rank = self.listnet_loss(lvis_pred_scores, lvis_target_scores)
            loss_rank = self.self_distillation_loss(lvis_pred_scores, lvis_target_scores)
        else:
            loss_rank = torch.tensor(0)
        
        losses = {
            "loss_triplet": loss_class,
            "loss_bg": loss_background,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_rank": loss_rank
        }
        return losses
