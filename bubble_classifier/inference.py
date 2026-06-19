from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "bubble_classifier" / "runs" / "resnet18" / "best_model.pt"
DEFAULT_FILLED_THRESHOLD = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BubblePrediction:
    image_path: str
    label: str
    confidence: float
    filled_probability: float | None
    blank_probability: float | None
    argmax_label: str
    argmax_confidence: float
    threshold: float

    def to_row(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "pred_label": self.label,
            "pred_confidence": self.confidence,
            "filled_probability": self.filled_probability,
            "blank_probability": self.blank_probability,
            "argmax_label": self.argmax_label,
            "argmax_confidence": self.argmax_confidence,
            "filled_threshold": self.threshold,
        }


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


class PilImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, Image.Image]], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_id, image = self.items[index]
        return self.transform(image.convert("RGB")), image_id


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_resnet18(class_count: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


class BubbleClassifier:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        filled_threshold: float = DEFAULT_FILLED_THRESHOLD,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= filled_threshold <= 1.0:
            raise ValueError("filled_threshold must be between 0 and 1")

        self.model_path = repo_path(model_path)
        self.filled_threshold = filled_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.class_names = list(checkpoint["class_names"])
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.image_size = int(checkpoint.get("image_size", 64))
        self.transform = build_eval_transform(self.image_size)

        self.model = build_resnet18(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path: str | Path) -> BubblePrediction:
        path = repo_path(image_path)
        dataset = ImagePathDataset([path], self.transform)
        return next(iter(self.predict_dataset(dataset, batch_size=1)))

    def predict_paths(
        self,
        image_paths: Iterable[str | Path],
        *,
        batch_size: int = 256,
    ) -> list[BubblePrediction]:
        paths = [repo_path(path) for path in image_paths]
        dataset = ImagePathDataset(paths, self.transform)
        return list(self.predict_dataset(dataset, batch_size=batch_size))

    def predict_images(
        self,
        images: Iterable[tuple[str, Image.Image]],
        *,
        batch_size: int = 256,
    ) -> list[BubblePrediction]:
        dataset = PilImageDataset(list(images), self.transform)
        return list(self.predict_dataset(dataset, batch_size=batch_size))

    @torch.inference_mode()
    def predict_dataset(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> Iterable[BubblePrediction]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for images, paths in loader:
            logits = self.model(images.to(self.device))
            probabilities = logits.softmax(dim=1).cpu()
            for path, scores in zip(paths, probabilities):
                yield self._prediction_from_scores(str(path), scores)

    def _prediction_from_scores(self, image_path: str, scores: torch.Tensor) -> BubblePrediction:
        argmax_index = int(scores.argmax().item())
        argmax_label = self.class_names[argmax_index]
        argmax_confidence = float(scores[argmax_index].item())

        filled_probability = self._probability(scores, "filled")
        blank_probability = self._probability(scores, "blank")

        if filled_probability is not None and blank_probability is not None:
            label = "filled" if filled_probability >= self.filled_threshold else "blank"
            confidence = filled_probability if label == "filled" else blank_probability
        else:
            label = argmax_label
            confidence = argmax_confidence

        return BubblePrediction(
            image_path=image_path,
            label=label,
            confidence=confidence,
            filled_probability=filled_probability,
            blank_probability=blank_probability,
            argmax_label=argmax_label,
            argmax_confidence=argmax_confidence,
            threshold=self.filled_threshold,
        )

    def _probability(self, scores: torch.Tensor, label: str) -> float | None:
        index = self.class_to_idx.get(label)
        if index is None:
            return None
        return float(scores[index].item())


def iter_image_paths(
    inputs: Iterable[str | Path],
    *,
    recursive: bool = False,
    extensions: set[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    image_paths: list[Path] = []
    for value in inputs:
        path = repo_path(value)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            image_paths.extend(
                child
                for child in path.glob(pattern)
                if child.is_file() and child.suffix.lower() in extensions
            )
        elif path.is_file():
            if path.suffix.lower() in extensions:
                image_paths.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(image_paths)
