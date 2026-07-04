import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


DATASETS = {
    "plant": {
        "data": "data-maize-plant-paper.yaml",
        "default_name": "maize_plant_paper",
        "classes": ["silk", "leaf", "root", "ears", "tassel", "up_ear_pos"],
        "imgsz": 768,
        "batch": 4,
        "epochs": 200,
        "patience": 50,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "amp": False,
        "close_mosaic": 0,
        "degrees": 5.0,
        "translate": 0.08,
        "scale": 0.25,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.25,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    "kernel": {
        "data": "data-maize-ears-paper.yaml",
        "default_name": "maize_kernel_paper",
        "classes": ["seed"],
        "imgsz": 1024,
        "batch": 4,
        "epochs": 120,
        "patience": 30,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "amp": True,
        "close_mosaic": 10,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-ready YOLO11-seg training entrypoint.")
    parser.add_argument(
        "--dataset",
        choices=["plant", "kernel", "both"],
        default="plant",
        help="Which paper dataset to train.",
    )
    parser.add_argument("--model", default="yolo11s-seg.pt", help="Pretrained YOLO11-seg weight path.")
    parser.add_argument("--project", default="runs/paper_seg", help="Output project directory.")
    parser.add_argument("--name", default=None, help="Run name override. If omitted, a dataset-specific name is used.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience override.")
    parser.add_argument("--optimizer", default=None, help="Optimizer name override.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Enable dataset caching.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="close_mosaic override.")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP override.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP override.")
    parser.set_defaults(amp=None)
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation override.")
    parser.add_argument("--translate", type=float, default=None, help="Translation augmentation override.")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation override.")
    parser.add_argument("--fliplr", type=float, default=None, help="Left-right flip probability override.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation override.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation override.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation override.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation override.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation override.")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste augmentation override.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run val/test on the current best.pt.",
    )
    return parser.parse_args()


def metrics_to_dict(metrics):
    if hasattr(metrics, "results_dict"):
        return dict(metrics.results_dict)
    return {}


def write_metrics(save_dir: Path, payload: dict):
    metrics_path = save_dir / "paper_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_path


def train_one(dataset_key: str, args):
    cfg = DATASETS[dataset_key]
    run_name = args.name or cfg["default_name"]
    imgsz = args.imgsz or cfg["imgsz"]
    epochs = args.epochs or cfg["epochs"]
    batch = args.batch or cfg["batch"]
    patience = args.patience or cfg["patience"]
    optimizer = args.optimizer or cfg["optimizer"]
    lr0 = args.lr0 if args.lr0 is not None else cfg["lr0"]
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg["weight_decay"]
    amp = args.amp if args.amp is not None else cfg["amp"]
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else cfg["close_mosaic"]
    degrees = args.degrees if args.degrees is not None else cfg["degrees"]
    translate = args.translate if args.translate is not None else cfg["translate"]
    scale = args.scale if args.scale is not None else cfg["scale"]
    fliplr = args.fliplr if args.fliplr is not None else cfg["fliplr"]
    hsv_h = args.hsv_h if args.hsv_h is not None else cfg["hsv_h"]
    hsv_s = args.hsv_s if args.hsv_s is not None else cfg["hsv_s"]
    hsv_v = args.hsv_v if args.hsv_v is not None else cfg["hsv_v"]
    mosaic = args.mosaic if args.mosaic is not None else cfg["mosaic"]
    mixup = args.mixup if args.mixup is not None else cfg["mixup"]
    copy_paste = args.copy_paste if args.copy_paste is not None else cfg["copy_paste"]

    print(f"\n=== Training dataset: {dataset_key} ===")
    print(f"data: {cfg['data']}")
    print(f"classes: {cfg['classes']}")
    print(f"model: {args.model}")
    print(f"project/name: {args.project}/{run_name}")
    print(f"imgsz={imgsz}, epochs={epochs}, batch={batch}, device={args.device}")
    print(f"optimizer={optimizer}, lr0={lr0}, weight_decay={weight_decay}, amp={amp}, close_mosaic={close_mosaic}")
    print(
        "augments="
        f"degrees={degrees}, translate={translate}, scale={scale}, fliplr={fliplr}, "
        f"hsv_h={hsv_h}, hsv_s={hsv_s}, hsv_v={hsv_v}, mosaic={mosaic}, mixup={mixup}, copy_paste={copy_paste}"
    )

    save_dir = Path(args.project) / run_name
    last_pt = save_dir / "weights" / "last.pt"

    if args.resume:
        if not last_pt.exists():
            raise FileNotFoundError(f"Cannot resume because last.pt was not found: {last_pt}")
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO(args.model)

    if not args.skip_train:
        model.train(
            data=cfg["data"],
            task="segment",
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=args.workers,
            device=args.device,
            patience=patience,
            seed=args.seed,
            pretrained=True,
            cache=args.cache,
            close_mosaic=close_mosaic,
            optimizer=optimizer,
            lr0=lr0,
            weight_decay=weight_decay,
            amp=amp,
            degrees=degrees,
            translate=translate,
            scale=scale,
            fliplr=fliplr,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            project=args.project,
            name=run_name,
            exist_ok=True,
            resume=args.resume,
            verbose=True,
        )

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=cfg["data"],
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        workers=max(1, min(args.workers, 2)),
    )
    test_metrics = best_model.val(
        data=cfg["data"],
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        workers=max(1, min(args.workers, 2)),
    )

    payload = {
        "dataset": dataset_key,
        "data_yaml": cfg["data"],
        "classes": cfg["classes"],
        "model": args.model,
        "best_pt": str(best_pt),
        "settings": {
            "imgsz": imgsz,
            "epochs": epochs,
            "batch": batch,
            "workers": args.workers,
            "device": args.device,
            "patience": patience,
            "optimizer": optimizer,
            "lr0": lr0,
            "weight_decay": weight_decay,
            "seed": args.seed,
            "cache": args.cache,
            "amp": amp,
            "close_mosaic": close_mosaic,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "fliplr": fliplr,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": copy_paste,
        },
        "val": metrics_to_dict(val_metrics),
        "test": metrics_to_dict(test_metrics),
    }
    metrics_path = write_metrics(save_dir, payload)

    print("\nValidation metrics:")
    print(json.dumps(payload["val"], indent=2, ensure_ascii=False))
    print("\nTest metrics:")
    print(json.dumps(payload["test"], indent=2, ensure_ascii=False))
    print(f"\nSaved metrics: {metrics_path}")


def main():
    args = parse_args()
    dataset_keys = ["plant", "kernel"] if args.dataset == "both" else [args.dataset]
    for dataset_key in dataset_keys:
        train_one(dataset_key, args)


if __name__ == "__main__":
    main()
