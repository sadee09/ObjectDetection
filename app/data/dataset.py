import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class AerialDataset(Dataset):
    """
    Dataset class for aerial image object detection
    Expects data structure:
    root_dir/
        ├── images/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── labels/
            ├── image1.txt
            ├── image2.txt
            └── ...
    """

    def __init__(self, root_dir, classes, transform=None, train=True, image_size=640):
        """
        Args:
            root_dir (str): Root directory path
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether this is training or validation dataset
            image_size (int): Size to resize images to (maintains aspect ratio with padding)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_size = image_size

        self.class_dict = {c: i for i, c in enumerate(classes)}

        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")

        self.images = []
        self.labels = []

        valid_extensions = (".jpg", ".jpeg", ".png")

        for filename in os.listdir(
            self.image_dir
        ):  # [: n] to train only on a subset of all data
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(self.image_dir, filename)
                label_path = os.path.join(
                    self.label_dir, os.path.splitext(filename)[0] + ".txt"
                )

                if os.path.exists(label_path) and os.path.exists(image_path):
                    self.images.append(image_path)
                    self.labels.append(label_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No valid images found in {root_dir}")

    def resize_image_and_boxes(self, image, boxes):
        """
        Resize image and adjust bounding boxes while maintaining aspect ratio
        Args:
            image (numpy.ndarray): Input image
            boxes (numpy.ndarray): Bounding boxes [x1, y1, x2, y2]
        Returns:
            tuple: (resized_image, adjusted_boxes)
        """
        orig_h, orig_w = image.shape[:2]

        scale = self.image_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        square_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        pad_h = (self.image_size - new_h) // 2
        pad_w = (self.image_size - new_w) // 2

        square_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = image

        if boxes is not None and len(boxes):
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            boxes[:, [0, 2]] += pad_w
            boxes[:, [1, 3]] += pad_h

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.image_size)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.image_size)

        return Image.fromarray(square_img), boxes

    def polygon_to_bbox(self, polygon):
        """
        Convert polygon to bounding box
        Args:
            polygon (list): List of normalized coordinates (x1, y1, x2, y2, ...)
        Returns:
            bbox (list): Bounding box [x_min, y_min, x_max, y_max]
        """
        coords = np.array(polygon).reshape(-1, 2)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return [x_min, y_min, x_max, y_max]

    def load_annotations(self, label_path):
        """
        Load annotations from text file and convert polygons to bounding boxes
        """
        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        labels = []

        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            polygon = parts[1:]

            bbox = self.polygon_to_bbox(polygon)

            bbox[0::2] = [coord * self.image_size for coord in bbox[0::2]]
            bbox[1::2] = [coord * self.image_size for coord in bbox[1::2]]

            if class_id in self.class_dict:
                boxes.append(bbox)
                labels.append(self.class_dict[class_id])

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx):
        """
        Get item by index
        Returns:
            tuple: (image, target) where target is a dictionary containing:
                - boxes (FloatTensor[N, 4]): bounding boxes in [x1, y1, x2, y2] format
                - labels (Int64Tensor[N]): labels for each box
        """
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = self.load_annotations(self.labels[idx])

        image, boxes = self.resize_image_and_boxes(image, boxes)

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target

    def __len__(self):
        return len(self.images)

    def get_height_and_width(self, idx):
        """
        Return height and width of the image
        Required by some detection models
        """
        return self.image_size, self.image_size
