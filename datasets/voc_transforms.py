from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _boxes_centers_xyxy(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    return cx, cy


def _clip_boxes_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w)
    boxes[:, 2] = boxes[:, 2].clamp(0, w)
    boxes[:, 1] = boxes[:, 1].clamp(0, h)
    boxes[:, 3] = boxes[:, 3].clamp(0, h)
    # Ensure x2>=x1, y2>=y1
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    boxes[:, 2] = torch.max(x1, x2)
    boxes[:, 3] = torch.max(y1, y2)
    return boxes


def _filter_boxes_min_size(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    min_wh: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_wh) & (h >= min_wh)
    return boxes[keep], labels[keep]


@dataclass
class SSDLightRandomCropAndResize:
    """
    Simplified SSD-style augmentation: ONLY random cropping (no photometric distortion).
    Boxes are kept if their center lies inside the crop.
    """

    output_size: Tuple[int, int]
    min_scale: float = 0.3
    max_scale: float = 1.0
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    max_trials: int = 50
    min_keep_boxes: int = 1
    min_box_wh: float = 1.0

    def __call__(
        self,
        image: Image.Image,
        boxes_xyxy: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        if boxes_xyxy.numel() == 0:
            # No boxes: just resize
            out_w, out_h = self.output_size[1], self.output_size[0]
            img = TF.resize(image, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
            return img, boxes_xyxy, labels

        w, h = image.size
        out_h, out_w = self.output_size

        best_fallback = None
        for _ in range(self.max_trials):
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

            crop_area = scale * w * h
            crop_w = int(round((crop_area * aspect) ** 0.5))
            crop_h = int(round((crop_area / aspect) ** 0.5))

            crop_w = max(1, min(crop_w, w))
            crop_h = max(1, min(crop_h, h))

            if crop_w <= 1 or crop_h <= 1:
                continue

            x0 = random.randint(0, w - crop_w)
            y0 = random.randint(0, h - crop_h)

            cx, cy = _boxes_centers_xyxy(boxes_xyxy)
            keep = (cx >= x0) & (cx <= x0 + crop_w) & (cy >= y0) & (cy <= y0 + crop_h)
            if keep.sum().item() < self.min_keep_boxes:
                continue

            boxes = boxes_xyxy[keep].clone()
            labs = labels[keep].clone()

            # Crop image
            img_c = TF.crop(image, top=y0, left=x0, height=crop_h, width=crop_w)

            # Adjust boxes into crop coords
            boxes[:, 0] -= x0
            boxes[:, 2] -= x0
            boxes[:, 1] -= y0
            boxes[:, 3] -= y0

            boxes = _clip_boxes_xyxy(boxes, crop_w, crop_h)
            boxes, labs = _filter_boxes_min_size(boxes, labs, min_wh=self.min_box_wh)

            # If everything got filtered out, try another crop.
            if boxes.numel() == 0:
                continue

            # Resize to output size
            img_out = TF.resize(img_c, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)

            sx = out_w / float(crop_w)
            sy = out_h / float(crop_h)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

            return img_out, boxes, labs

        # Fallback: no crop (resize original)
        # Scale boxes accordingly.
        img_out = TF.resize(image, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
        sx = out_w / float(w)
        sy = out_h / float(h)
        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes, labels = _filter_boxes_min_size(boxes, labels, min_wh=self.min_box_wh)
        return img_out, boxes, labels


@dataclass
class ModifiedYOLOJitterRandomResize:
    """
    Modified YOLO-style augmentation:
    - Geometric: expand (scale up) then random crop back to original size (smaller targets).
    - Photometric: color jitter, with jitter range changed to [-0.3, 0.1].
    - Optional horizontal flip.

    This is a practical approximation of the described idea:
    SSD tends to crop inside image and enlarges objects; YOLO aug compensates by using
    a larger patch context, plus a stronger/weaker jitter range.
    """

    output_size: Tuple[int, int]
    expand_ratio_range: Tuple[float, float] = (1.0, 1.2)
    jitter_range: Tuple[float, float] = (-0.3, 0.1)
    flip_prob: float = 0.5
    max_trials: int = 50
    min_keep_boxes: int = 1
    min_box_wh: float = 1.0

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        # Factors around 1.0 so negative jitter becomes "darker" but not invalid.
        j0, j1 = self.jitter_range
        brightness = 1.0 + random.uniform(j0, j1)
        contrast = 1.0 + random.uniform(j0, j1)
        saturation = 1.0 + random.uniform(j0, j1)
        hue = random.uniform(j0, j1)  # torchvision hue shift uses [-0.5, 0.5]

        img = TF.adjust_brightness(image, brightness_factor=brightness)
        img = TF.adjust_contrast(img, contrast_factor=contrast)
        img = TF.adjust_saturation(img, saturation_factor=saturation)
        img = TF.adjust_hue(img, hue_factor=hue)
        return img

    def __call__(
        self,
        image: Image.Image,
        boxes_xyxy: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        w, h = image.size
        out_h, out_w = self.output_size

        if boxes_xyxy.numel() == 0:
            img_out = TF.resize(image, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
            img_out = self._apply_color_jitter(img_out)
            if random.random() < self.flip_prob:
                img_out = TF.hflip(img_out)
            return img_out, boxes_xyxy, labels

        # Expand: scale up then crop back.
        expand = random.uniform(self.expand_ratio_range[0], self.expand_ratio_range[1])
        nw = max(1, int(round(w * expand)))
        nh = max(1, int(round(h * expand)))

        img_exp = TF.resize(image, [nh, nw], interpolation=TF.InterpolationMode.BILINEAR)
        sx0 = nw / float(w)
        sy0 = nh / float(h)
        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] *= sx0
        boxes[:, [1, 3]] *= sy0

        # Random crop back to original size (W,H)
        for _ in range(self.max_trials):
            x0 = random.randint(0, max(0, nw - w))
            y0 = random.randint(0, max(0, nh - h))

            cx, cy = _boxes_centers_xyxy(boxes)
            keep = (cx >= x0) & (cx <= x0 + w) & (cy >= y0) & (cy <= y0 + h)
            if keep.sum().item() < self.min_keep_boxes:
                continue

            boxes_c = boxes[keep].clone()
            labs_c = labels[keep].clone()
            img_c = TF.crop(img_exp, top=y0, left=x0, height=h, width=w)

            boxes_c[:, 0] -= x0
            boxes_c[:, 2] -= x0
            boxes_c[:, 1] -= y0
            boxes_c[:, 3] -= y0

            boxes_c = _clip_boxes_xyxy(boxes_c, w, h)
            boxes_c, labs_c = _filter_boxes_min_size(boxes_c, labs_c, min_wh=self.min_box_wh)
            if boxes_c.numel() == 0:
                continue

            # Color jitter on the cropped image
            img_c = self._apply_color_jitter(img_c)

            # Optional flip
            if random.random() < self.flip_prob:
                img_c = TF.hflip(img_c)
                x1 = boxes_c[:, 0].clone()
                x2 = boxes_c[:, 2].clone()
                boxes_c[:, 0] = w - x2
                boxes_c[:, 2] = w - x1

            # Resize to output
            img_out = TF.resize(img_c, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
            sx = out_w / float(w)
            sy = out_h / float(h)
            boxes_c[:, [0, 2]] *= sx
            boxes_c[:, [1, 3]] *= sy
            return img_out, boxes_c, labs_c

        # Fallback: just jitter + (optional flip) then resize
        img_c = TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
        img_c = self._apply_color_jitter(img_c)
        boxes_f = boxes_xyxy.clone()
        labs_f = labels.clone()
        if random.random() < self.flip_prob:
            img_c = TF.hflip(img_c)
            x1 = boxes_f[:, 0].clone()
            x2 = boxes_f[:, 2].clone()
            boxes_f[:, 0] = w - x2
            boxes_f[:, 2] = w - x1

        img_out = TF.resize(img_c, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
        sx = out_w / float(w)
        sy = out_h / float(h)
        boxes_f[:, [0, 2]] *= sx
        boxes_f[:, [1, 3]] *= sy
        boxes_f, labs_f = _filter_boxes_min_size(boxes_f, labs_f, min_wh=self.min_box_wh)
        return img_out, boxes_f, labs_f


@dataclass
class RandomChoiceSSDOrModifiedYOLO:
    """
    Randomly choose between:
    - simplified SSD crop (no photometric distortion)
    - modified YOLO jitter (expand/crop + jitter with [-0.3, 0.1])
    """

    output_size: Tuple[int, int]
    ssd_p: float = 0.5

    def __post_init__(self) -> None:
        self._ssd = SSDLightRandomCropAndResize(output_size=self.output_size)
        self._yolo = ModifiedYOLOJitterRandomResize(output_size=self.output_size)

    def __call__(
        self,
        image: Image.Image,
        boxes_xyxy: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        if random.random() < self.ssd_p:
            return self._ssd(image, boxes_xyxy, labels)
        return self._yolo(image, boxes_xyxy, labels)

