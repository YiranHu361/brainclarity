import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as T
from sklearn.metrics import classification_report, f1_score
from torchvision.datasets import ImageFolder


def find_conv2d_out_shape(hin: int, win: int, conv: nn.Conv2d, pool: int = 2) -> Tuple[int, int]:
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = (hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    wout = (win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)


class CNN_TUMOR(nn.Module):
    def __init__(self, shape_in=(3, 256, 256), initial_filters=8, num_fc1=100, dropout_rate=0.25, num_classes=2):
        super().__init__()

        cin, hin, win = shape_in
        init_f = initial_filters

        self.conv1 = nn.Conv2d(cin, init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(hin, win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv4)

        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flatten)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x


def get_dataloaders(data_dir: Path, batch_size: int = 32, val_split: float = 0.2):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(30),
            T.ToTensor(),
            normalize,
        ]
    )
    val_tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), normalize])

    full_dataset = ImageFolder(str(data_dir), transform=train_tf)
    num_total = len(full_dataset)
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    train_set, val_set = data.random_split(full_dataset, [num_train, num_val])
    # override transform for val set to deterministic
    val_set.dataset.transform = val_tf

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return full_dataset.classes, full_dataset.class_to_idx, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

    return running_loss / total, running_corrects / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_labels = []
    all_preds = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    avg_acc = running_corrects / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, digits=3)
    return avg_loss, avg_acc, f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="public/Brain Tumor Data Set")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="public/model")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data dir not found: {data_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes, class_to_idx, train_loader, val_loader = get_dataloaders(
        data_dir, batch_size=args.batch_size, val_split=0.2
    )
    model = CNN_TUMOR(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    if best_f1 > 0:
        model.load_state_dict(best_state)

    _, _, _, report = evaluate(model, val_loader, criterion, device)
    print("Validation report:\n", report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "brain_tumor_state_dict.pt")
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    model.eval()
    onnx_path = output_dir / "brain_tumor.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved ONNX model to {onnx_path}")

    meta = {"classes": classes, "class_to_idx": class_to_idx}
    with open(output_dir / "class_map.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata with classes: {classes}")


if __name__ == "__main__":
    main()
