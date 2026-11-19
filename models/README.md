# Whole-plant models

| File | Use |
| --- | --- |
| [best.pt](best.pt) | YOLO11s-seg, validation-selected epoch 105; checkpoint used for the reported test metrics |
| [last.pt](last.pt) | Final checkpoint after 155 epochs |
| [plant_deployment_5class.onnx](plant_deployment_5class.onnx) | ONNX deployment model, input 1920x1920 |

Class IDs: 0 leaf, 1 root, 2 ears, 3 silk, 4 tassel. The upper-ear leaf is selected during post-processing.
The model was retrained after loss of the previous PT checkpoint. The authors confirmed that
PyTorch evaluation of best.pt and ONNX deployment used the same model weights. last.pt is
the final training checkpoint. See the [dated confirmation](../docs/author_confirmation_2026-11-19.md)
and [file hashes](model_manifest.json).

[whole_plant_training_archive.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/whole_plant_training_archive.zip) includes the 256 images,
original/converted annotations, frozen split, training settings, evaluator and test evidence.
Checkpoints retain applicable Ultralytics/upstream licensing; the image-data license does not replace it.

The 256-image model-development collection has no variety overlap with the 150-variety phenotype
panel. The reported AP and class-union IoU were measured using best.pt at a maximum input side of
1024 pixels; no independent ONNX performance evaluation is claimed. Format-specific export metadata
and hashes are retained. See [validation details](../docs/model_notes.md).
