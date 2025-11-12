"""
一个用于演示的小型前馈神经网络训练脚本。

特点：
- 使用纯 PyTorch 实现，默认在 CPU 上训练，避免 CUDA 依赖。
- 随机生成可分数据集（两个高斯团簇），适合快速 sanity check。
- 每个 epoch 输出一行 JSON，方便被 SciAgent 捕获到 `metrics.jsonl`。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class ToyConfig:
    input_dim: int = 2
    hidden_dim: int = 32
    dropout: float = 0.0
    num_classes: int = 2
    samples: int = 2000
    train_split: float = 0.8


class GaussianToyDataset(Dataset):
    """构造两个高斯团簇的二分类玩具数据集。"""

    def __init__(self, total_samples: int, input_dim: int) -> None:
        half = total_samples // 2
        mean_a = torch.ones(input_dim) * -2.0
        mean_b = torch.ones(input_dim) * 2.0

        cov = torch.eye(input_dim)
        data_a = torch.distributions.MultivariateNormal(mean_a, cov).sample((half,))
        data_b = torch.distributions.MultivariateNormal(mean_b, cov).sample((total_samples - half,))

        self.features = torch.cat([data_a, data_b], dim=0)
        self.labels = torch.cat(
            [torch.zeros(half, dtype=torch.long), torch.ones(total_samples - half, dtype=torch.long)],
            dim=0,
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy MLP trainer for SciAgent demo.")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden-dim", type=int, default=32, help="隐藏层尺寸")
    parser.add_argument("--dropout", type=float, default=0.0, help="隐藏层 dropout 比例")
    parser.add_argument("--device", type=str, default="cpu", help="设备（cpu/cuda）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    config = ToyConfig(hidden_dim=args.hidden_dim, dropout=args.dropout)
    dataset = GaussianToyDataset(config.samples, config.input_dim)

    train_len = int(config.samples * config.train_split)
    val_len = config.samples - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = SimpleClassifier(config.input_dim, config.hidden_dim, config.dropout, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            running_acc += accuracy(logits.detach(), targets) * targets.size(0)

        train_loss = running_loss / train_len
        train_acc = running_acc / train_len

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                val_acc += accuracy(logits, targets) * targets.size(0)

        val_loss /= val_len
        val_acc /= val_len

        result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "device": str(device),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


