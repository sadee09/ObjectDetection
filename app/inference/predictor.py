import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


class Predictor:
    def __init__(self, model, config):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.classes = config["data"]["classes"]
        self.confidence_threshold = config["model"]["confidence_threshold"]
        self.image_size = config["data"]["image_size"]

    def preprocess_image(self, image):
        """Preprocess image to match training format"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        original_size = image.shape[:2]

        scale = self.image_size / max(original_size)
        new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        square_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        pad_h = (self.image_size - new_h) // 2
        pad_w = (self.image_size - new_w) // 2
        square_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = image

        return {
            "image": F.to_tensor(square_img),
            "original_size": original_size,
            "scale": scale,
            "pad": (pad_w, pad_h),
        }

    def adjust_predictions(self, predictions, original_size, scale, pad):
        """Adjust predictions back to original image size"""
        for pred in predictions:
            bbox = pred["bbox"]
            bbox[0] = bbox[0] - pad[0]
            bbox[2] = bbox[2] - pad[0]
            bbox[1] = bbox[1] - pad[1]
            bbox[3] = bbox[3] - pad[1]

            bbox[0] = bbox[0] / scale
            bbox[2] = bbox[2] / scale
            bbox[1] = bbox[1] / scale
            bbox[3] = bbox[3] / scale

            bbox[0] = max(0, min(bbox[0], original_size[1]))
            bbox[2] = max(0, min(bbox[2], original_size[1]))
            bbox[1] = max(0, min(bbox[1], original_size[0]))
            bbox[3] = max(0, min(bbox[3], original_size[0]))

            pred["bbox"] = [float(x) for x in bbox]

        return predictions

    @torch.no_grad()
    def predict(self, image_bytes):
        processed = self.preprocess_image(image_bytes)
        image = processed["image"].to(self.device)

        outputs = self.model([image])

        print(outputs)

        predictions = []
        for output in outputs:
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            for box, score, label in zip(boxes, scores, labels):
                predictions.append(
                    {
                        "label": self.classes[label],
                        "confidence": float(score),
                        "bbox": [float(x) for x in box],
                    }
                )

        predictions = self.adjust_predictions(
            predictions,
            processed["original_size"],
            processed["scale"],
            processed["pad"],
        )

        return predictions
