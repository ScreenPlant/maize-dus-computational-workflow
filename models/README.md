# Whole-plant models

| File | Use |
| --- | --- |
| [best.pt](best.pt) | New YOLO11s-seg, validation-selected epoch 105; use for the reported new test metrics |
| [last.pt](last.pt) | Final checkpoint after 155 epochs |
| [plant_deployment_5class.onnx](plant_deployment_5class.onnx) | Separate historical deployment artifact, input 1920x1920 |

Class IDs: 0 leaf, 1 root, 2 ears, 3 silk, 4 tassel. The upper-ear leaf is selected during post-processing.
The new PT was retrained after loss of the historical PT; it is not a recovery of that file.
New PT metrics do not characterize the historical ONNX. See [validation](../docs/model_notes.md)
and [file hashes](model_manifest.json).

[whole_plant_training_archive.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/whole_plant_training_archive.zip) includes the 256 images,
original/converted annotations, frozen split, training settings, evaluator and test evidence.
Checkpoints retain applicable Ultralytics/upstream licensing; the image-data license does not replace it.
