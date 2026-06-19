"""Dataset, model, and metric helpers for layout detector training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


MASK_CHANNELS = ("page_mask", "marker_heatmap", "bubble_heatmap", "red_grid_mask")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(path_text: str | Path, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base / path


class LayoutDataset(Dataset):
    """Reads generated layout_v0 images and masks from manifest.jsonl."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: str,
        image_size: tuple[int, int],
        channels: tuple[str, ...] = MASK_CHANNELS,
        limit: int = 0,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.channels = channels

        manifest_path = dataset_dir / "manifest.jsonl"
        records = [record for record in read_manifest(manifest_path) if record["split"] == split]
        if limit > 0:
            records = records[:limit]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        output_width, output_height = self.image_size

        image_path = self.dataset_dir / record["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)
        image_tensor = (image_tensor - 0.5) / 0.5

        masks = []
        for channel in self.channels:
            mask_path = self.dataset_dir / record["mask_paths"][channel]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_AREA)
            masks.append(mask.astype(np.float32) / 255.0)
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0))

        meta = {
            "sample_id": record["sample_id"],
            "base_image_id": record["base_image_id"],
            "split": record["split"],
            "image_path": str(image_path),
            "label_path": str(self.dataset_dir / record["label_path"]),
            "source_width": int(record.get("source_width", 0) or 0),
            "source_height": int(record.get("source_height", 0) or 0),
        }
        return image_tensor, mask_tensor, meta


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LayoutUNet(nn.Module):
    """Small U-Net for mask and heatmap prediction."""

    def __init__(self, *, out_channels: int, base_channels: int = 24) -> None:
        super().__init__()
        base = base_channels
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))
        x = self.up3(bottleneck, enc3)
        x = self.up2(x, enc2)
        x = self.up1(x, enc1)
        return self.out(x)


def build_layout_model(
    *,
    out_channels: int = len(MASK_CHANNELS),
    base_channels: int = 24,
) -> nn.Module:
    return LayoutUNet(out_channels=out_channels, base_channels=base_channels)


def parse_float_list(value: str, expected_count: int) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} comma-separated values, got {len(values)}")
    return values


def layout_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weights: tuple[float, ...],
    dice_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(targets)
    for channel_index, pos_weight in enumerate(pos_weights):
        weights[:, channel_index] += targets[:, channel_index] * (float(pos_weight) - 1.0)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * weights).mean()

    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    mse = F.mse_loss(probs, targets)
    return bce + dice_weight * dice + mse_weight * mse


@torch.no_grad()
def pixel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    channels: tuple[str, ...] = MASK_CHANNELS,
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    metrics: dict[str, dict[str, float]] = {}
    for index, channel in enumerate(channels):
        pred = preds[:, index]
        target = (targets[:, index] >= threshold).float()
        intersection = (pred * target).sum().item()
        pred_sum = pred.sum().item()
        target_sum = target.sum().item()
        dice = (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)
        metrics[channel] = {
            "dice": float(dice),
            "mse": float(F.mse_loss(probs[:, index], targets[:, index]).item()),
        }
    return metrics


def merge_metric_sums(
    totals: dict[str, dict[str, float]],
    batch_metrics: dict[str, dict[str, float]],
    *,
    weight: int,
) -> None:
    for channel, values in batch_metrics.items():
        channel_totals = totals.setdefault(channel, {})
        for name, value in values.items():
            channel_totals[name] = channel_totals.get(name, 0.0) + float(value) * weight


def finalize_metric_sums(
    totals: dict[str, dict[str, float]],
    *,
    total_weight: int,
) -> dict[str, dict[str, float]]:
    if total_weight <= 0:
        return {}
    return {
        channel: {name: value / total_weight for name, value in values.items()}
        for channel, values in totals.items()
    }


def mean_channel_metric(metrics: dict[str, dict[str, float]], metric_name: str) -> float:
    values = [channel_metrics[metric_name] for channel_metrics in metrics.values()]
    return float(sum(values) / max(len(values), 1))


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.clip((image * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0)
