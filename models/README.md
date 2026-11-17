# Whole-plant models

| File | Status | Input size |
| --- | --- | --- |
| [plant_deployment_5class.onnx](plant_deployment_5class.onnx) | Existing five-class study deployment model | 1920 × 1920 |
| [plant_later_6class.pt](plant_later_6class.pt) | Newly supplied `best.pt`, confirmed to be a later six-class version | Saved training setting: 768 × 768 |

Both files are stored directly in this branch. Use GitHub's download/raw option or obtain the branch with Git. The later checkpoint is 20,538,340 bytes and has not been modified. `last.pt` is retained locally and is not part of this submission.

The model files serve outdoor whole-plant segmentation. Study traits T26/T27 use the deployment workflow; T28 is derived. The other 20 traits use indoor conventional processing.

| ID | Five-class deployment ONNX | Later checkpoint |
| --- | --- | --- |
| 0 | leaf | silk |
| 1 | root | leaf |
| 2 | ears | root |
| 3 | silk | ears |
| 4 | tassel | tassel |
| 5 | — | up_ear_pos |

The later checkpoint is not an asserted export source of the five-class ONNX. Training provenance and accuracy evidence are distinguished in [docs/yolo11_validation.md](../docs/yolo11_validation.md). File sizes and SHA-256 checksums are recorded in `model_manifest.json`.

From the repository root, using the recorded inference environment:

```powershell
python scripts/predict_plant_seg.py --model models/plant_deployment_5class.onnx --input private_inputs/images --output outputs/plant_5class --imgsz 1920
python scripts/predict_plant_seg.py --model models/plant_later_6class.pt --input private_inputs/images --output outputs/plant_later_6class --imgsz 768
```

These commands perform segmentation, not model training or complete raw-image trait extraction. Dataset images and annotations remain pending upload.
