from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "bubble_classifier/data/manifest.csv"
DEFAULT_OUTPUT_DIR = "bubble_classifier/runs/resnet18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 bubble state classifier.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print training progress every N batches. Use 0 to disable batch logs.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class BubbleDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], transform) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(repo_path(row["crop_path"])).convert("RGB")
        return self.transform(image), self.class_to_idx[row["gt_label"]]


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return train_transform, eval_transform


def build_model(class_count: int, *, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def make_train_loader(dataset: BubbleDataset, rows: list[dict], args: argparse.Namespace) -> DataLoader:
    if args.no_balanced_sampler:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

    counts = Counter(row["gt_label"] for row in rows)
    weights = [1.0 / counts[row["gt_label"]] for row in rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    *,
    epoch: int,
    progress_every: int,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(loader)
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

        if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
            print(
                f"epoch={epoch:03d} batch={batch_index}/{batch_count} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}",
                flush=True,
            )

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, class_names: list[str]) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * targets.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        for target, pred in zip(targets.cpu(), preds.cpu()):
            confusion[target, pred] += 1

    per_class = {}
    for index, name in enumerate(class_names):
        row_total = int(confusion[index].sum().item())
        per_class[name] = {
            "support": row_total,
            "recall": float(confusion[index, index].item() / row_total) if row_total else None,
        }

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> int:
    args = parse_args()
    manifest_path = repo_path(args.manifest)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise SystemExit("manifest must contain train and val rows")

    class_names = sorted({row["gt_label"] for row in train_rows})
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = BubbleDataset(train_rows, class_to_idx, train_transform)
    val_dataset = BubbleDataset(val_rows, class_to_idx, eval_transform)
    train_loader = make_train_loader(train_dataset, train_rows, args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"manifest={manifest_path}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"device={device}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_name={torch.cuda.get_device_name(0)}", flush=True)
    print(f"classes={class_names}", flush=True)
    print(f"train_count={len(train_rows)} val_count={len(val_rows)}", flush=True)
    print(
        f"batch_size={args.batch_size} workers={args.workers} "
        f"balanced_sampler={not args.no_balanced_sampler}",
        flush=True,
    )

    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"starting_epoch={epoch:03d}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            progress_every=args.progress_every,
        )
        print(f"evaluating_epoch={epoch:03d}", flush=True)
        val_metrics = evaluate(model, val_loader, criterion, device, class_names)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
            ,
            flush=True,
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

    summary = {
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "device": str(device),
        "class_names": class_names,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "best_val_accuracy": best_accuracy,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"best_model={output_dir / 'best_model.pt'}")
    print(f"metrics={output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
