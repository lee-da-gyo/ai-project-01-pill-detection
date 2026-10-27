"""2단계 모델의 2단계 – 알약 crop 분류기 (VGG16 / ResNet50 / EfficientNet-B0).

왜 2단계인가 (E0 결론)
- 검출기는 위치를 거의 완벽하게 찾는다. 남은 문제는 "약 이름" 판별이다.
- crop 단위 분류기는 알약 하나만 보므로 각인·색·모양 같은 세부 특징에 집중할 수 있고,
  회전·색 변화 같은 강한 증강을 박스 라벨 걱정 없이 줄 수 있다. 희귀 약은 가중 샘플링으로 자주 보여 준다.

가이드의 "VGG / ResNet 등 CNN 모델 비교, 정확도·정밀도·재현율로 최적 모델 선정"을 이 모듈로 수행한다.
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARCHS = {
    "vgg16": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def crop_boxes(gt: pd.DataFrame, img_dir: Path, pad: float = 0.1) -> list[Image.Image]:
    """박스 영역을 pad 비율만큼 넓혀 자른다. 검출 박스가 조금 어긋나도 알약이 잘리지 않게 한다."""
    crops = []
    for f, g in gt.groupby("file_name", sort=False):
        img = Image.open(Path(img_dir) / f).convert("RGB")
        W, H = img.size
        for r in g.itertuples():
            px, py = r.w * pad, r.h * pad
            crops.append(img.crop((max(0, r.x - px), max(0, r.y - py), min(W, r.x + r.w + px), min(H, r.y + r.h + py))))
    return crops


def ordered(gt: pd.DataFrame) -> pd.DataFrame:
    """crop_boxes와 같은 순서(파일별로 묶음)로 행을 정렬한다."""
    order = {f: i for i, f in enumerate(dict.fromkeys(gt["file_name"]))}
    return gt.assign(_o=gt["file_name"].map(order)).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def build_transforms(size: int = 224):
    train = transforms.Compose([
        transforms.RandomRotation(180, fill=(120, 130, 150)),  # 위에서 찍은 알약은 방향이 자유롭다 (배경과 비슷한 색으로 채움)
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train, val


class CropDataset(Dataset):
    def __init__(self, crops: list[Image.Image], labels: list[int], transform):
        self.crops, self.labels, self.transform = crops, labels, transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        return self.transform(self.crops[i]), self.labels[i]


def build_model(arch: str, num_classes: int) -> nn.Module:
    fn, weights = ARCHS[arch]
    m = fn(weights=weights)
    if arch == "vgg16":
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif arch == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


@torch.no_grad()
def predict_proba(model: nn.Module, crops: list[Image.Image], transform, device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        x = torch.stack([transform(c) for c in crops[i:i + batch]]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(torch.softmax(model(x).float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def classification_metrics(y_true, prob: np.ndarray) -> dict:
    """정확도·macro 정밀도/재현율/F1 + log loss.

    log loss(정답 클래스 확률의 -log 평균)는 정확도가 같을 때의 판별 기준이다. 맞히더라도 확신이 낮으면 커진다.
    """
    y_true = np.asarray(y_true)
    y_pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    nll = float(-np.log(np.clip(prob[np.arange(len(y_true)), y_true], 1e-7, 1)).mean())
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r, "macro_f1": f, "log_loss": nll}


def train_classifier(arch: str, train_crops, train_labels, val_crops, val_labels, num_classes: int, out_dir: Path,
                     epochs: int = 30, batch: int = 32, lr: float = 3e-4, size: int = 224, seed: int = 42,
                     label_smoothing: float = 0.1, log=print) -> dict:
    """학습하면서 val macro F1이 가장 좋은 가중치를 저장한다. 에포크별 기록과 최고 지표를 반환한다."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_train, tf_val = build_transforms(size)

    # 희귀 클래스를 자주 뽑는다: 샘플 가중치 ∝ 1/√빈도
    counts = np.bincount(train_labels, minlength=num_classes)
    w = 1 / np.sqrt(counts[train_labels])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(train_labels), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(CropDataset(train_crops, train_labels, tf_train), batch_size=batch, sampler=sampler, num_workers=0)

    model = build_model(arch, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train_labels) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"macro_f1": -1, "log_loss": float("inf")}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
        prob = predict_proba(model, val_crops, tf_val, device)
        m = classification_metrics(val_labels, prob)
        history.append({"epoch": ep, "train_loss": total / len(train_labels), **m})
        # F1이 더 높거나, 같으면 log loss가 더 낮은 에포크를 저장한다
        if m["macro_f1"] > best["macro_f1"] or (m["macro_f1"] == best["macro_f1"] and m["log_loss"] < best["log_loss"]):
            best = {"epoch": ep, **m}
            torch.save(model.state_dict(), out_dir / "best.pt")
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log(f"  [{arch}] epoch {ep:2d}/{epochs} loss {total / len(train_labels):.3f} | "
                f"acc {m['accuracy']:.3f} P {m['macro_precision']:.3f} R {m['macro_recall']:.3f} F1 {m['macro_f1']:.3f} "
                f"logloss {m['log_loss']:.3f}")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"arch": arch, "params_M": round(params, 1), "train_min": round((time.time() - t0) / 60, 1), **best}


def load_classifier(arch: str, num_classes: int, weights: Path, device=None) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(arch, num_classes)
    m.load_state_dict(torch.load(weights, map_location=device))
    return m.to(device).eval()
