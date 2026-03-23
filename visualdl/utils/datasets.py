# visualdl/utils/datasets.py


# Standard imports
import itertools
import os

# External imports
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import numpy as np
import torch

# Constants
ALLOWED_DATASET_FILE_FORMATS = tuple(
    {
        ".bmp",
        ".dib",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jp2",
        ".png",
        ".webp",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pxm",
        ".pnm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
        ".exr",
        ".hdr",
        ".pic",
    }
)
"""See https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56 for more information."""


class ClassificationDataset(Dataset):
    def __init__(self, root_folder, transform, class_weights=False):
        super().__init__()
        self.root_folder = root_folder
        self.transform = transform
        self.num_classes = len(os.listdir(root_folder))
        self.classes = os.listdir(root_folder)
        if class_weights:
            tmp = [
                1 - y
                for y in [
                    x
                    / sum(
                        [
                            len(os.listdir(os.path.join(root_folder, folder)))
                            for folder in os.listdir(root_folder)
                        ]
                    )
                    for x in [
                        len(os.listdir(os.path.join(root_folder, folder)))
                        for folder in os.listdir(root_folder)
                    ]
                ]
            ]
            self.class_weights = [z + (1 - max(tmp)) for z in tmp]
        self.image_class_tuple = list(
            itertools.chain.from_iterable(
                [
                    [
                        (
                            os.path.join(path, file),
                            self.classes.index(os.path.split(path)[-1]),
                        )
                        for file in files
                        if file.endswith(ALLOWED_DATASET_FILE_FORMATS)
                    ]
                    for path, directories, files in os.walk(root_folder)
                ]
            )
        )

    def __len__(self):
        return len(self.image_class_tuple)

    def __getitem__(self, idx):
        img_path, label = self.image_class_tuple[idx]
        img = io.imread(img_path).astype(np.float32)
        transform = self.transform
        if transform:
            img = transform(image=img)["image"]
        img = img / 255.0
        return torch.tensor(img, dtype=torch.float).permute(2, 0, 1), torch.tensor(
            label, dtype=torch.long
        )


class InstanceSegmentationDataset(Dataset):
    def __init__(self, folder, transform=None, use_cache=True):
        self.folder = folder
        # Build paired lists by matching basenames (stem) between images and masks
        images_dir = os.path.join(self.folder, "images")
        masks_dir = os.path.join(self.folder, "labels")
        img_candidates = [
            f
            for f in os.listdir(images_dir)
            if f.endswith(ALLOWED_DATASET_FILE_FORMATS) or f.endswith(".npy")
        ]
        mask_candidates = [
            f for f in os.listdir(masks_dir) if f.endswith(ALLOWED_DATASET_FILE_FORMATS)
        ]
        # Map stem -> filename
        def stem(name: str) -> str:
            return os.path.splitext(name)[0]
        img_map = {stem(f): f for f in img_candidates}
        mask_map = {stem(f): f for f in mask_candidates}
        common = sorted(set(img_map.keys()) & set(mask_map.keys()))
        self.train_images = [os.path.join(images_dir, img_map[s]) for s in common]
        self.train_masks = [os.path.join(masks_dir, mask_map[s]) for s in common]
        self.transform = transform
        self.use_cache = use_cache
        self.cached_data = []
        to_delete = []
        self.max_boxes = 0
        self.max_label = 0
        for im, la in zip(self.train_images, self.train_masks):
            mask = cv2.imread(la, 0)  # Currently changed to the original image
            self.max_label = max(np.max(mask), self.max_label)
            if np.count_nonzero(mask) == 0:
                to_delete.append((im, la))
        for im, la in to_delete:
            self.train_images.remove(im)
            self.train_masks.remove(la)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        if self.train_images[idx].endswith(".npy"):
            img = np.load(self.train_images[idx]).astype(np.float32)
        else:
            img = io.imread(self.train_images[idx]).astype(np.float32)
        img = img / 255.0
        mask = io.imread(self.train_masks[idx], as_gray=True)
        # Use the following line if you want to use binary masks, required for some online datasets
        # mask[mask > 0] = 1.0
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        tmp = mask.copy().astype(np.uint8)
        tmp[tmp > 0] = 1

        contours, hierachy = cv2.findContours(
            tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        labels = []
        masks = []

        for cnt, cont in enumerate(contours):
            rect = cv2.boundingRect(cont)
            xmin, ymin, xmax, ymax = (
                rect[0],
                rect[1],
                rect[0] + rect[2],
                rect[1] + rect[3],
            )
            rect = (xmin, ymin, xmax, ymax)
            mask2 = np.zeros_like(tmp)
            cv2.drawContours(mask2, [cont], -1, 255, -1)
            pts = np.where(mask2 > 0)
            _cls = mask[pts[0], pts[1]]
            cls = np.unique(_cls)[
                np.argmax(np.unique(_cls, return_counts=True)[1])
            ]  # - 1
            mask2[pts[0], pts[1]] = mask[pts[0], pts[1]]
            boxes.append(rect)
            labels.append(cls.astype(np.int64))
            mask2[mask2 > 0] = 1  # somehow needs to be binary
            masks.append(mask2.astype(np.uint8))
        if len(contours) != 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        if len(contours) != 0:
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels = torch.as_tensor([0], dtype=torch.int64)
        if len(contours) != 0:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.as_tensor(torch.empty(0, 512, 512), dtype=torch.uint8)
        if len(contours) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0], dtype=torch.float32)
        is_crowd = torch.zeros((len(contours),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = dict()
        self.max_boxes = max(self.max_boxes, len(contours))
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd
        # self.cached_data.append() TODO: probably add to cached data but poor RAM :(
        return torch.tensor(img, dtype=torch.float).permute(2, 0, 1), target


class SegmentationDataset(Dataset):
    def __init__(self, folder, transform, class_weights=False):
        self.folder = folder
        self.transform = transform
        # Build paired lists by matching basenames (stem) between images and masks
        images_dir = os.path.join(self.folder, "images")
        masks_dir = os.path.join(self.folder, "labels")
        img_candidates = [
            f
            for f in os.listdir(images_dir)
            if f.endswith(ALLOWED_DATASET_FILE_FORMATS) or f.endswith(".npy")
        ]
        mask_candidates = [
            f for f in os.listdir(masks_dir) if f.endswith(ALLOWED_DATASET_FILE_FORMATS)
        ]
        # Map stem -> filename
        def stem(name: str) -> str:
            return os.path.splitext(name)[0]
        img_map = {stem(f): f for f in img_candidates}
        mask_map = {stem(f): f for f in mask_candidates}
        common = sorted(set(img_map.keys()) & set(mask_map.keys()))
        self.train_images = [os.path.join(images_dir, img_map[s]) for s in common]
        self.train_masks = [os.path.join(masks_dir, mask_map[s]) for s in common]
        if class_weights:
            vals = {}
            mask_loader = tqdm(self.train_masks)
            mask_loader.set_description("Calculating class weights")
            for image in mask_loader:
                img = io.imread(image, as_gray=True)
                # img[img > 0] = 1
                unique, counts = np.unique(img, return_counts=True)
                su = float(sum(counts))

                for val, cnt in zip(unique, counts):
                    if not val in vals:
                        vals[val] = cnt / su
                    else:
                        vals[val] += cnt / su
            final = []
            for val in vals.values():
                final.append(1 - (val / len(self.train_masks)))
            self.class_weights = final

            print(f"Calculated class weights:{self.class_weights}")
        assert len(self.train_images) == len(self.train_masks)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        if self.train_images[idx].endswith(".npy"):
            img = np.load(self.train_images[idx]).astype(np.float32)
        else:
            img = io.imread(self.train_images[idx]).astype(np.float32)
        mask = io.imread(self.train_masks[idx], as_gray=True).astype(np.float32)
        # Use the following line if you want to use binary masks, required for some online datasets
        # mask[mask > 0] = 1.0
        img = img / 255.0
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return torch.tensor(img, dtype=torch.float).permute(2, 0, 1), torch.tensor(
            mask, dtype=torch.long
        )


class ImageOnlyDataset(Dataset):
    """
    Dataset without any labels, just plain images for inference.

    Args:
        Dataset ():
    """

    def __init__(self, folder):
        self.folder = folder
        self.images = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if x.endswith(ALLOWED_DATASET_FILE_FORMATS)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = io.imread(self.images[idx]).astype(np.float32)
        img = img / 255.0
        return torch.tensor(img, dtype=torch.float)
