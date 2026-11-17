# YOLO11-seg: method and validation

## Role in the study

YOLO11-seg is used only to segment outdoor whole-plant images supporting upper-ear insertion height (T26) and plant length (T27). The study application combines segmentation with geometric measurement. T28 is the ratio of the corresponding variety means and has no separate image acquisition. The remaining 20 traits use conventional processing of indoor images, without additional trained segmentation models.

The study workflow above follows the authors' confirmed acquisition and processing scope. The presence of additional routines in retained application source does not establish that those routines were used for the reported traits.

## Models and provenance

| Record | Existing deployment ONNX | Newly supplied later checkpoint |
| --- | --- | --- |
| File | `models/plant_deployment_5class.onnx` | `models/plant_later_6class.pt` |
| Role | Retained study deployment model | Later version with an added `up_ear_pos` class, confirmed by the authors |
| Classes, in ID order | leaf, root, ears, silk, tassel | silk, leaf, root, ears, tassel, up_ear_pos |
| Model record | Segmentation ONNX; fixed 1920 × 1920 input | YOLO11s-seg; `SegmentationModel` with a six-class `Segment` head |
| Recorded Ultralytics | 8.3.32 in export metadata | 8.3.89 in checkpoint metadata |
| Original training PyTorch version | [Pending verification]; ONNX producer 2.5.1 is an export record | [Pending verification] |
| Source training images and annotation inventory | [Pending verification] | [Pending verification] |
| Annotation tool, procedure and class definitions | [Pending verification] | [Pending verification]; class names are read directly from the checkpoint |
| Train/val/test membership, counts and grouping | [Pending verification] | [Pending verification]; the saved `split=val` argument does not identify member images |
| Overlap with the 150-variety panel | [Pending verification] | [Pending verification] |
| Held-out mask mAP50 / mAP50-95 | [Pending verification] | [Pending verification] |
| Mask IoU and its aggregation definition | [Pending verification] | [Pending verification] |

The later checkpoint is not presented as the source of the existing five-class ONNX. Its class count, order and confirmed development stage differ. Export linkage requires the matching original checkpoint and export record.

## Recorded parameters of the later checkpoint

These are saved configuration values, not a reconstruction of the original five-class training run. `epochs` is the configured limit; the stripped checkpoint reports `epoch=-1`, which does not identify the selected epoch.

| Parameter | Saved value |
| --- | --- |
| `model` | `yolo11s-seg.pt` |
| `task` | `segment` |
| `epochs` | `200` |
| `batch` | `2` |
| `imgsz` | `768` |
| `optimizer` | `AdamW` |
| `lr0` | `0.0005` |
| `seed` | `42` |
| `amp` | `False` |
| `deterministic` | `True` |
| `pretrained` | `True` |
| `patience` | `50` |
| `split` | `val` |

The checkpoint refers to `data-maize-plant-paper.yaml`; its original contents and associated data remain [Pending verification]. The old YAML notice in the repository is not the recovered training manifest.

The later checkpoint contains saved validation metric fields: mask mAP50 = **0.76179** and mask mAP50-95 = **0.39667**. These are checkpoint-recorded validation values. They have not been recomputed here and are not an independent test-set result, pixel mean IoU, or validation of the five-class study model. Matching logs, annotations and split membership are still required for a reproducible validation report.

## Checks and pending validation

Local checks load the supplied weights, inspect their segmentation head and run inference without training or changing the files. Execution compatibility is separate from accuracy. The current software versions are listed in [environment/README.md](../environment/README.md).

To complete validation, the original model-specific data manifest, annotations and evaluation records are required. Report mask AP at the stated IoU thresholds and define how any separately reported Mask IoU is calculated. No missing score, split or overlap determination is inferred from trait-level agreement or fingerprint resampling.

## Grain-row number (T32): conventional processing

T32 uses a lateral indoor ear image, not YOLO segmentation. The inspected retained C++ functions are `Maize101Ear`, `GetLocalMaxima` and `MaizeEarColsRows` in `Process1Maize.cpp`.

1. The conventional branch masks the ear ROI, converts it to grayscale, applies adaptive mean thresholding and extracts candidate kernel centers using Gaussian smoothing and local maxima.
2. The candidate nearest the ear-contour centroid is selected as a starting point. `MaizeEarColsRows` traces candidate centers to the left and right, choosing the nearest admissible next point. Its transverse chain uses the source's scale-dependent checks: horizontal step at least `0.2 * DotsPerCM` and vertical offset at most `0.4 * DotsPerCM`.
3. The number of points in this transverse chain is `row_num`; chains shorter than three are set to zero. The actual output statement is `ret_str.push_back("32:" + to_string(row_num * 2));`.

Thus, the output doubles the estimated visible transverse count to approximate the circumferential row count. It assumes that the visible side represents approximately half the circumference and that the chosen chain adequately represents the rows. A single view does not directly observe the hidden side. Perspective, occlusion and irregular rows can violate the approximation, and the routine produces even-valued individual estimates.

The retained source includes an optional ear-AI branch, but the authors confirm it was disabled for the reported 2025 ear analysis. Limited cross-section images were not matched-ear correction or validation evidence. No retrospective replacement of the T32 observations is claimed. The C++ measurement application is not publicly released in this repository.
