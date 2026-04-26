# ===- pytorch-lenet-train.py --------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# LeNet training on MNIST (PyTorch). 默认短时训练并打印测试准确率；可用 ``--epochs``
# 拉长训练。保存的整模型与 ``buddy-lenet-import.py`` / TTIR 流程兼容。
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeNet on MNIST; save full model for Buddy import.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="torch.save 路径（默认：本目录 lenet-model.pth）。",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output
    if out is None:
        out = Path(__file__).resolve().parent / "lenet-model.pth"

    torch.manual_seed(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cpu")
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- train loss: {running / len(train_loader):.4f}"
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test accuracy: {acc:.2f}% ({correct}/{total})")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"Saved model to {out.resolve()}")


if __name__ == "__main__":
    main()
