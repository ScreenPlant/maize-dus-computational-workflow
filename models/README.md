# Plant segmentation model

`plant_deployment_5class.onnx` is the trained whole-plant model used for deployment. Its input size is 1920 × 1920 pixels.

| Class ID | Label |
| --- | --- |
| 0 | leaf |
| 1 | root |
| 2 | ears |
| 3 | silk |
| 4 | tassel |

The upper-ear leaf is selected during post-processing; it is not a separate model class. The 2025 ear and kernel analyses used conventional image processing.

```powershell
python scripts/predict_plant_seg.py --model models/plant_deployment_5class.onnx --input private_inputs/images --output outputs/segmentation
```

The file checksum is recorded in `model_manifest.json`. Training records and evaluation scope are described in [docs/model_notes.md](../docs/model_notes.md).
