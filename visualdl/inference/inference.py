import segmentation_models_pytorch as smp

# from torchvision.ops.misc import Conv2d
from torch import load
from ..utils.model_utils import (
    predict_images,
    make_single_class_per_contour,
    predict_instance_segmentation,
    predict_classification_images,
)
import torch
import numpy as np
import os
import cv2
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import square
from ..models.dpt.models import DPTSegmentationModel
from ..models.dpt.models import DPTSegmentationModel
from ..models.custom import U2NET
from ..models.TransInUnet import TransInUnet
from ..models.hrnet import HRNetV2
from ..models.caranet.CaraNet import caranet
from ..models.doubleunet.doubleunet import DoubleUnet
from ..models.unetr import UNETR
from ..trainer.series.series_trainer import get_model, predict_series
from ..trainer.mlp.mlp_trainer import get_mlp_model, predict_mlp
import timm
from ..trainer.instance.utils import GeneralizedRCNNTransform


def predict_od(model, imgs, confidence=0.45):
    size = imgs[0].shape[0]
    model.conf = confidence
    preds = model(imgs, size=size)
    finals = []
    for cnt, img in enumerate(imgs):
        tmp = []
        boxes = preds.xyxy[cnt]
        for box in boxes:
            middlex = int(box[0] + (box[2] - box[0]) / 2)
            middley = int(box[1] + (box[3] - box[1]) / 2)
            data = list(box.detach().cpu().numpy())
            data.append((middlex, middley))
            tmp.append(tuple(data))
        finals.append(tmp)
    return finals


class ModelInference:
    def __init__(
        self,
        weight_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        type="segmentation",
        watershed_od="",
    ):
        self.device = device
        self.type = type

        if type == "segmentation":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            self.model = eval(state["model"].replace('"imagenet"', "None"))
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            self.has_distance_map = False
            if "has_distance_map" in self.state:
                self.has_distance_map = bool(self.state["has_distance_map"])

        elif type == "od":
            dirname = os.path.dirname(__file__)
            dirname = dirname.replace(dirname.split("\\")[-1], "dependencies/yolov5")
            self.state = {}
            self.model, self.state["custom_data"] = torch.hub.load(
                dirname,
                "custom",
                path=weight_path,
                source="local",
                return_custom_data=True,
            )

        elif type == "segmentation_od":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            self.model = eval(state["model"].replace('"imagenet"', "None"))
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            self.has_distance_map = False
            if "has_distance_map" in self.state:
                self.has_distance_map = bool(self.state["has_distance_map"])

            dirname = os.path.dirname(__file__)
            dirname = dirname.replace(dirname.split("\\")[-1], "dependencies/yolov5")
            self.od_state = {}
            self.model_od, self.od_state["custom_data"] = torch.hub.load(
                dirname,
                "custom",
                path=watershed_od,
                source="local",
                return_custom_data=True,
            )

        elif type == "instance":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            if not "pretrained_backbone=False" in state["model"].replace(
                "\n", ""
            ).replace(" ", ""):
                self.model = eval(
                    state["model"]
                    .replace("\n", "")
                    .replace(" ", "")
                    .replace(")", ", pretrained_backbone=False)")
                )
            else:
                self.model = eval(state["model"].replace("\n", "").replace(" ", ""))
            if "in_channels" in state and state["in_channels"] != 3:
                self.model.transform = GeneralizedRCNNTransform(
                    image_mean=[0.485, 0.456, 0.406],
                    image_std=[0.229, 0.224, 0.225],
                    min_size=(800,),
                    max_size=1333,
                    mode="bilinear",
                )
            if "in_channels" in state:
                self.model.backbone.body.conv1 = torch.nn.Conv2d(
                    state["in_channels"],
                    self.model.backbone.body.conv1.out_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )

                # self.model.roi_heads.mask_predictor.mask_fcn_logits = Conv2d(256, self.state['nc'], 1)
            self.model.eval()
            self.model.load_state_dict(state["model_state_dict"])

        elif type == "series":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            self.model = get_model(
                state["classes"],
                state["continous"],
                state["features"],
                use_lstm=state["use_lstm"] if "use_lstm" in state else True,
            )
            self.model.eval()
            self.model.load_state_dict(state["model_state_dict"])
        elif type == "classification":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            print(state["model"])
            self.model = eval(state["model"])
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()

        elif type == "mlp":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device("cpu"))
            else:
                state = load(weight_path)
            self.state = state
            self.model = get_mlp_model(state["in_features"], state["out_features"])
            self.model.eval()
            self.model.load_state_dict(state["model_state_dict"])
        else:
            raise ValueError(f"Unknown modeltype provided: {type}")

    def __call__(self, images):
        """
        Calls predict of this instance selected model
        """
        return self.predict(images)

    def predict(
        self,
        images,
        single_class_per_contour=False,
        min_size=None,
        confidence=0.45,
        fill_holes=False,
        mlp_output_type="default",
    ):
        if self.type == "segmentation":
            return predict_images(
                self.model,
                images,
                self.device,
                single_class_per_contour,
                min_size,
                self.has_distance_map,
                fill_holes=fill_holes,
            )
        elif self.type == "od":
            return predict_od(self.model, images, confidence=confidence)
        elif self.type == "segmentation_od":
            all_segmentations = []
            boxes = predict_od(self.model_od, images, confidence=confidence)
            maps = predict_images(
                self.model,
                images,
                self.device,
                single_class_per_contour,
                min_size,
                self.has_distance_map,
                fill_holes=fill_holes,
            )[0]
            # images must be rgb
            for image, box, map in zip(images, boxes, maps):
                label_map = np.int32(np.zeros_like(map))
                p = 1
                for b in box:
                    label_map = cv2.circle(label_map, b[-1], 1, p, -1)
                    p += 1

                # ndi watershed
                distance = ndi.distance_transform_edt(map)
                # mapss = np.uint8(ndi.binary_fill_holes(map))
                labels = watershed(-distance, label_map, mask=map, watershed_line=True)
                labels[labels > 0] = map[labels > 0]
                kernel = np.ones((2, 2), np.uint8)
                labels = cv2.erode(np.uint8(labels), kernel)

                # rgb_mask = cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                # markers = cv2.watershed(rgb_mask, label_map)
                # empty = np.zeros_like(markers).astype(np.uint8)
                # empty[markers == -1] = 255
                # kernel = np.ones((2, 2), np.uint8)
                # labels = cv2.dilate(empty, kernel)
                # map[labels == 255] = 0
                # labels = map.copy()
                all_segmentations.append(
                    make_single_class_per_contour(labels, min_size)
                )
            return all_segmentations
        elif self.type == "instance":
            return predict_instance_segmentation(
                self.model, images, self.device, confidence
            )
        elif self.type == "series":
            if "multi_label" in self.state:
                return predict_series(
                    self.model,
                    images,
                    self.device,
                    self.state["classes"],
                    self.state["continous"],
                    self.state["multi_label"],
                )
            else:
                return predict_series(
                    self.model,
                    images,
                    self.device,
                    self.state["classes"],
                    self.state["continous"],
                    multi_label=False,
                )
        elif self.type == "classification":
            return predict_classification_images(self.model, images, self.device)
        elif self.type == "mlp":
            range_predictions = (
                0
                if "range_predictions" not in self.state.keys()
                else self.state["range_predictions"]
            )
            return predict_mlp(
                self.model, images, self.device, mlp_output_type, range_predictions
            )
