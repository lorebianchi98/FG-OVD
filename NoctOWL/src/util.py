from collections import defaultdict
from typing import Dict
import time
import numpy as np
import re
from tabulate import tabulate
import torch

from transformers import OwlViTForObjectDetection, Owlv2ForObjectDetection, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.ops import box_convert as _box_convert
from torchvision.utils import draw_bounding_boxes
from datetime import timedelta




class GeneralLossAccumulator:
    def __init__(self):
        self.loss_values = defaultdict(lambda: 0)
        self.n = 0

    def update(self, losses: Dict[str, torch.tensor]):
        for k, v in losses.items():
            self.loss_values[k] += v.item()
        self.n += 1

    def get_values(self):
        averaged = {}
        for k, v in self.loss_values.items():
            averaged[k] = round(v / self.n, 5)
        return averaged

    def reset(self):
        self.value = 0


class ProgressFormatter:
    def __init__(self):
        self.table = {
            "epoch": [],
            "class loss": [],
            "bg loss": [],
            "box loss": [],
            "map": [],
            "map@0.5": [],
            "map (L/M/S)": [],
            "mar (L/M/S)": [],
            "time elapsed": [],
        }
        self.start = time.time()

    def update(self, epoch, train_metrics, val_metrics):
        self.table["epoch"].append(epoch)
        self.table["class loss"].append(train_metrics["loss_triplet"])
        self.table["bg loss"].append(train_metrics["loss_bg"])
        self.table["box loss"].append(
            train_metrics["loss_bbox"] + train_metrics["loss_giou"]
        )
        self.table["map"].append(round(val_metrics["map"].item(), 3))
        self.table["map@0.5"].append(round(val_metrics["map_50"].item(), 3))

        map_s = round(val_metrics["map_small"].item(), 2)
        map_m = round(val_metrics["map_medium"].item(), 2)
        map_l = round(val_metrics["map_large"].item(), 2)

        self.table["map (L/M/S)"].append(f"{map_l}/{map_m}/{map_s}")

        mar_s = round(val_metrics["mar_small"].item(), 2)
        mar_m = round(val_metrics["mar_medium"].item(), 2)
        mar_l = round(val_metrics["mar_large"].item(), 2)

        self.table["mar (L/M/S)"].append(f"{mar_l}/{mar_m}/{mar_s}")

        self.table["time elapsed"].append(
            timedelta(seconds=int(time.time() - self.start))
        )

    def print(self):
        print()
        print(tabulate(self.table, headers="keys"))
        print()


class BoxUtil:
    @classmethod
    def scale_bounding_box(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        imwidth: torch.tensor, # [M] 
        imheight: torch.tensor, # [M] 
        mode: str,  # up | down
    ):
        # expanding to perform the batched resize
        imwidth = imwidth.unsqueeze(1).unsqueeze(2) 
        imheight = imheight.unsqueeze(1).unsqueeze(2)
        if mode == "down":
            boxes_batch[:, :, (0, 2)] /= imwidth
            boxes_batch[:, :, (1, 3)] /= imheight
            return boxes_batch
        elif mode == "up":
            boxes_batch[:, :, (0, 2)] *= imwidth
            boxes_batch[:, :, (1, 3)] *= imheight
            return boxes_batch

    @classmethod
    def draw_box_on_image(
        cls,
        image: str or torch.tensor,  # cv2 image
        boxes_batch: torch.tensor,
        labels_batch: list = None,
        color=(0, 255, 0),
    ):
        if isinstance(image, str):
            image = read_image(image)
        if labels_batch is None:
            for _boxes in boxes_batch:
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, width=2)
        else:
            for _boxes, _labels in zip(boxes_batch, labels_batch):
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, _labels, width=2)
        return image

    # see https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
    @classmethod
    def box_convert(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        in_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
        out_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
    ):
        return _box_convert(boxes_batch, in_format, out_format)


class ModelUtil:
    @classmethod
    def compare_models(cls, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')
    
    @classmethod
    def copy_model(cls, src_model, dst_model):
        """
        Copy the parameters of the trained model inside the base model, in order to have a configuration callable from huggingface
        """
        
        # Take the pretrained components that are useful to us
        if isinstance(dst_model, OwlViTForObjectDetection):
            dst_model.owlvit.vision_model = src_model.backbone
        elif isinstance(dst_model, Owlv2ForObjectDetection):
            dst_model.owlv2.vision_model = src_model.backbone
        else:
            raise ValueError("The loaded pretrained model is invalid")

        dst_model.layer_norm = src_model.post_post_layernorm
        # self.class_predictor = PatchedOwlViTClassPredictionHead(
        #     pretrained_model.class_head
        # )
        dst_model.class_head.dense0 = src_model.class_predictor.dense0
        dst_model.box_head = src_model.box_head 
        dst_model.compute_box_bias = src_model.compute_box_bias
        dst_model.sigmoid = src_model.sigmoid
        
        return dst_model
    
    @classmethod
    def create_base_model(cls, src_model, base_config, device='cuda'):
        if 'owlvit' in base_config:
            model = OwlViTForObjectDetection.from_pretrained(base_config)
        elif 'owlv2' in base_config:
            model = Owlv2ForObjectDetection.from_pretrained(base_config)
        else:
            raise ValueError("The starting configuration must come from from owlvit or owlv2")
        model = cls.copy_model(src_model, model)
        model.to(device)
        return model
    
def to_x1y1x2y2(boxes):
    cx, cy, w, h = boxes[0, :, 0], boxes[0, :, 1], boxes[0, :, 2], boxes[0, :, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    new_boxes = torch.stack((x1, y1, x2, y2), dim=1).unsqueeze(dim=0)
    return new_boxes

def get_processor(base_model):
    if 'owlvit' in base_model:
        base_model = base_model.replace("owlvit", "owlv2")
        size = {"height":768, "width":768} if 'base' in base_model else {"height": 840, "width": 840}
        return AutoProcessor.from_pretrained(base_model, size=size)
    elif 'owlv2' in base_model:
        return AutoProcessor.from_pretrained(base_model)
    else:
        raise Exception("Invalid base model")
        


def process_single_string(s: str) -> str:
    NOT_PROMPTABLE_MARKER = '#'
    s = s.lower()

    # Remove all characters that are not either alphanumeric, or dash, or space, or NOT_PROMPTABLE_MARKER
    s = re.sub(f'[^a-z0-9-{NOT_PROMPTABLE_MARKER} ]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'-+', '-', s)
    s = s.strip()

    # Remove characters that equal the promptability-marker but appear somewhere other than the start of the string
    s = re.sub(f'([^^]){NOT_PROMPTABLE_MARKER}+', r'\1', s)

    return s