# Computational Workflow and Analysis Scripts

## Purpose

This document summarizes the public computational workflow released with the maize DUS phenotyping study. It is intended to help readers understand how the segmentation models, inference utilities, phenotype-analysis scripts, and variety-level fingerprint analyses are organized in this repository.

## Scope of the public release

The public repository focuses on the reproducible computational components directly supporting the manuscript:

1. YOLOv11seg training for whole-plant organ segmentation and kernel instance segmentation.
2. ONNX export and deployment-oriented inference.
3. Annotation-conversion utilities for segmentation labels.
4. Trait-analysis, clustering, and visualization scripts.
5. Variety-level core fingerprint construction, subset search, and robustness validation.

This repository is a reproducibility-oriented release rather than a complete laboratory working directory. Large historical runs, full private datasets, and unrelated exploratory scripts are intentionally excluded from the public-facing workflow.

## Workflow overview

The study-level computational workflow is organized into six stages:

1. Annotation preparation and label conversion.
2. YOLOv11seg training and validation.
3. ONNX export and deployment-oriented inference.
4. ROI-supported phenotypic trait extraction and trait-matrix assembly.
5. Trait filtering, clustering, and visualization.
6. Variety-level core fingerprint scoring, subset optimization, and robustness assessment.

## Stage 1. Annotation preparation and label conversion

### LabelMe to YOLO segmentation labels

The annotation-conversion entry point is:

- [scripts/labelme_to_yolo_seg.py](../scripts/labelme_to_yolo_seg.py)

This script converts LabelMe polygon annotations into YOLO segmentation label files and is used to prepare training labels for YOLOv11seg.

### LabelMe to masked image crops

The mask-generation helper is:

- [scripts/labelme_to_mask.py](../scripts/labelme_to_mask.py)

This script generates mask-based crops from LabelMe annotations for inspection and preprocessing.

## Stage 2. YOLOv11seg training and validation

### Final training entry point

The main manuscript-facing training script is:

- [scripts/paper_train_seg.py](../scripts/paper_train_seg.py)

It supports:

- `plant`: six-class whole-plant organ segmentation
- `kernel`: single-class kernel instance segmentation
- `both`: sequential execution of both tasks

The final dataset configurations are:

- [configs/data-maize-plant-paper.yaml](../configs/data-maize-plant-paper.yaml)
- [configs/data-maize-ears-paper.yaml](../configs/data-maize-ears-paper.yaml)

### Final paper-ready runs

The paper-ready retraining runs included in this release are:

- Whole-plant model:
  - [results/paper_seg/maize_plant_paper_aug_b2/paper_metrics.json](../results/paper_seg/maize_plant_paper_aug_b2/paper_metrics.json)
- Kernel model:
  - [results/paper_seg/maize_kernel_paper_v3/paper_metrics.json](../results/paper_seg/maize_kernel_paper_v3/paper_metrics.json)

These files provide the validation and test metrics used to summarize model quality in the manuscript.

## Stage 3. ONNX export and deployment

### ONNX export

The export helper is:

- [scripts/seg2onnx.py](../scripts/seg2onnx.py)

This script exports trained segmentation checkpoints to ONNX format for deployment-oriented use.

### Inference

The inference helper is:

- [scripts/predict_seg.py](../scripts/predict_seg.py)

This script runs segmentation inference on individual images or image folders and supports annotated-output and mask-output generation.

## Stage 4. Trait extraction and phenotype-table assembly

In the public release, the ROI-to-trait layer is represented as an integration point between segmentation outputs and downstream tabular analysis. Whole-plant organ masks are used to define ROIs for plant architecture and color-related traits, whereas kernel masks are used to support kernel arrangement, grain-row estimation, kernel size, and kernel-color-related descriptors.

The complete image-measurement layer is not yet released as a single standalone public module. The present repository therefore emphasizes the segmentation, analysis, and fingerprint layers that directly support the computational results described in the manuscript.

## Stage 5. Trait analysis, clustering, and visualization

The downstream trait-analysis layer is represented by:

- [scripts/trait_selection_hierarchical_clustering.py](../scripts/trait_selection_hierarchical_clustering.py)
- [scripts/dus_type_clustering_pipeline.py](../scripts/dus_type_clustering_pipeline.py)
- [scripts/core_trait_pca_kmeans_pipeline.py](../scripts/core_trait_pca_kmeans_pipeline.py)
- [scripts/analyze_table3.py](../scripts/analyze_table3.py)

These scripts support trait filtering, normalization, feature ranking, clustering, dimensionality reduction, and visualization.

## Stage 6. Variety-level core fingerprint workflow and outputs

### Main analysis scripts

The core scripts for the variety-level fingerprint workflow are:

- [scripts/build_variety_core_fingerprint.py](../scripts/build_variety_core_fingerprint.py)
- [scripts/compare_core_fingerprint_schemes.py](../scripts/compare_core_fingerprint_schemes.py)
- [scripts/internal_validate_core_fingerprint.py](../scripts/internal_validate_core_fingerprint.py)
- [scripts/plot_global_phenotype_heatmap.py](../scripts/plot_global_phenotype_heatmap.py)
- [scripts/analyze_category_fingerprint.py](../scripts/analyze_category_fingerprint.py)
- [scripts/build_mixed_fingerprint.py](../scripts/build_mixed_fingerprint.py)

### Key outputs

Representative fingerprint outputs included in this repository are stored in:

- [outputs/variety_core_fingerprint](../outputs/variety_core_fingerprint)

This directory includes:

- processed encoded trait matrices
- phenotypic ID tables
- core-trait scoring summaries
- trait-subset comparison tables
- internal validation and robustness summaries

The most manuscript-facing outputs include:

- [core_trait_scoring_table.csv](../outputs/variety_core_fingerprint/core_trait_scoring_table.csv)
- [selected_core_traits.csv](../outputs/variety_core_fingerprint/selected_core_traits.csv)
- [encoded_trait_matrix_for_id.csv](../outputs/variety_core_fingerprint/encoded_trait_matrix_for_id.csv)
- [phenotypic_id_codes.csv](../outputs/variety_core_fingerprint/phenotypic_id_codes.csv)
- [objective_scheme_comparison.csv](../outputs/variety_core_fingerprint/objective_scheme_comparison.csv)
- [trait_number_comparison.csv](../outputs/variety_core_fingerprint/trait_number_comparison.csv)
- [internal_validation_overview.csv](../outputs/variety_core_fingerprint/internal_validation_overview.csv)

## Public data and model placeholders

This repository includes lightweight placeholders and documentation for:

- [data/README.md](../data/README.md)
- [models/README.md](../models/README.md)

Small public demo subsets can be placed under `data/demo/`, while deployable public model files can be placed under `models/` if redistribution is permitted.

## Recommended public repository structure

```text
maize-dus-computational-workflow/
  README.md
  LICENSE
  CITATION.cff
  configs/
  scripts/
  data/
  models/
  results/
  outputs/
  docs/
```

## Example commands

### Train the whole-plant model

```powershell
python scripts/paper_train_seg.py --dataset plant --name maize_plant_paper_aug_b2 --device 0 --cache
```

### Train the kernel model

```powershell
python scripts/paper_train_seg.py --dataset kernel --name maize_kernel_paper_v3 --device 0 --cache --batch 2 --lr0 0.0005 --no-amp --imgsz 640
```

### Run inference

```powershell
python scripts/predict_seg.py --help
```

### Build the variety-level fingerprint outputs

```powershell
python scripts/build_variety_core_fingerprint.py --help
```

## Summary

The public release centers on the paper-ready segmentation training pipeline, ONNX export and inference helpers, annotation-conversion tools, and downstream phenotype-analysis and fingerprint scripts. Together, these files provide the main computational workflow needed to interpret and reproduce the study-level results reported in the manuscript.
