import random, json, cv2
from pathlib import Path
from tqdm import tqdm
import torch, torchvision
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

Path("weights").mkdir(exist_ok=True)   

# ───────────────────────── Dataset ─────────────────────────
class CocoSet(torch.utils.data.Dataset):
    def __init__(self, img_root, ann_file):
        self.img_root = Path(img_root)
        data = json.load(open(ann_file))
        self.imgs = [d for d in data["images"]]
        self.anns = {}
        for a in data["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
      return len(self.imgs)

    def __getitem__(self, idx):
        info = self.imgs[idx]
        img = cv2.imread(str(self.img_root / info["file_name"]))
        if img is None:
            raise FileNotFoundError(f"missing {info['file_name']}")
        img = torch.as_tensor(img[:, :, ::-1] / 255, dtype=torch.float32).permute(2, 0, 1)

        annos = self.anns.get(info["id"], [])
        if annos:
            boxes = torch.tensor([a["bbox"] for a in annos], dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]     # xywh → xyxy
            labels = torch.tensor([a["category_id"] for a in annos], dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),     dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}

def collate(b): return tuple(zip(*b))

# ───────────────────────── Training ────────────────────────
def main():
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = CocoSet("data/images/train", "data/train.coco.json")
    val_set   = CocoSet("data/images/val",   "data/val.coco.json")

    tr_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate)
    va_loader = DataLoader(val_set,   batch_size=4, shuffle=False,num_workers=4, collate_fn=collate)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEV)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 5).to(DEV)

    for p in model.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-3, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()

    for epoch in range(4, 6):
        if epoch == 4:                             
            for p in model.backbone.parameters(): p.requires_grad = True
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-3, momentum=0.9, weight_decay=1e-4)
            print("🔓 Backbone unfrozen, lr=1e-3")

        model.train()
        pbar = tqdm(tr_loader, ncols=90, desc=f"Epoch {epoch}/5")
        for imgs, tgts in pbar:
            imgs = [i.to(DEV) for i in imgs]
            tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            with autocast("cuda"):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        torch.save(model.state_dict(), f"weights/frcnn_epoch{epoch}.pth")
        print(f"✅ saved weights/frcnn_epoch{epoch}.pth")

if __name__ == "__main__":
    main()