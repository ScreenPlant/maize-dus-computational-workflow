# Maize DUS Phenotyping Computational Workflow

This repository contains the computational workflow and analysis scripts supporting a maize DUS phenotyping study. It integrates instance segmentation, ROI-based phenotyping, trait-matrix analysis, and variety-level phenotypic fingerprint construction.

The released codebase is organized as a reproducibility-oriented repository rather than a full lab working directory. It focuses on the scripts, configurations, audit records, and selected outputs required to understand and reproduce the main computational components of the study.

## Study scope

This repository supports four major components of the study:

1. YOLOv11seg training and evaluation for maize whole-plant organ segmentation.
2. YOLOv11seg training and evaluation for maize-ear kernel instance segmentation.
3. Trait-level downstream analysis, clustering, and visualization.
4. Variety-level core fingerprint construction, subset search, and robustness assessment.

## Workflow overview

The overall computational workflow can be summarized as follows:

1. Annotation preparation and label conversion.
2. YOLOv11seg training and validation for whole-plant and kernel tasks.
3. ONNX export and deployment-oriented inference.
4. ROI-supported phenotypic trait extraction and trait-matrix assembly.
5. Trait filtering, clustering, and visualization.
6. Variety-level core fingerprint scoring, subset optimization, and robustness validation.

Detailed workflow notes are provided in:

- [docs/GitHub_Submission_Computational_Workflow.md](docs/GitHub_Submission_Computational_Workflow.md)

## Repository structure

```text
maize-dus-computational-workflow/
  README.md
  requirements.txt

  configs/
    data-maize-plant-paper.yaml
    data-maize-ears-paper.yaml

  scripts/
    paper_train_seg.py
    seg2onnx.py
    predict_seg.py
    labelme_to_yolo_seg.py
    labelme_to_mask.py
    trait_selection_hierarchical_clustering.py
    dus_type_clustering_pipeline.py
    core_trait_pca_kmeans_pipeline.py
    build_variety_core_fingerprint.py
    compare_core_fingerprint_schemes.py
    internal_validate_core_fingerprint.py
    rebuild_internal_validation_s7.py
    refresh_core_scoring_from_s3_s5.py
    build_supplementary_table_s6.py
    plot_main_representative_trait_agreement.py
    plot_supplementary_figure_s2_complete_agreement.py
    plot_supplementary_figure_s3_biological_discrimination.py
    regenerate_fig5c_from_current_table7.py
    plot_global_phenotype_heatmap.py
    analyze_table3.py
    analyze_category_fingerprint.py
    build_mixed_fingerprint.py

  results/
    paper_seg/
      maize_plant_paper_aug_b2/
        paper_metrics.json
      maize_kernel_paper_v3/
        paper_metrics.json

  outputs/
    variety_core_fingerprint/
      core_trait_scoring_table.csv
      selected_core_traits.csv
      encoded_trait_matrix_for_id.csv
      phenotypic_id_codes.csv
      objective_scheme_comparison.csv
      trait_number_comparison.csv
      internal_validation_overview.csv

  docs/
    GitHub_Submission_Computational_Workflow.md
    model_validation_summary.csv

  models/
    README.md

  data/
    README.md
    demo/
```

## Environment

Install the main Python dependencies with:

```powershell
pip install -r requirements.txt
```

Core dependencies include:

- `ultralytics`
- `torch`
- `onnxruntime`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `umap-learn`

## Segmentation training

The main paper-ready training entry point is:

- [scripts/paper_train_seg.py](scripts/paper_train_seg.py)

### Whole-plant model

Train the whole-plant organ segmentation model:

```powershell
python scripts/paper_train_seg.py --dataset plant --name maize_plant_paper_aug_b2 --device 0 --cache
```

### Kernel model

Train the kernel instance-segmentation model:

```powershell
python scripts/paper_train_seg.py --dataset kernel --name maize_kernel_paper_v3 --device 0 --cache --batch 2 --lr0 0.0005 --no-amp --imgsz 640
```

### Resume an interrupted run

```powershell
python scripts/paper_train_seg.py --dataset plant --name maize_plant_paper_aug_b2 --device 0 --cache --resume
```

### Evaluate an existing checkpoint without retraining

```powershell
python scripts/paper_train_seg.py --dataset kernel --name maize_kernel_paper_v3 --device 0 --skip-train
```

## ONNX export and inference

### Export a trained segmentation model to ONNX

```powershell
python scripts/seg2onnx.py
```

### Run segmentation inference

```powershell
python scripts/predict_seg.py
```

The current inference helper preserves the manuscript-era logic, including mask visualization and mask-only output generation, and now exposes its main inputs and outputs through CLI arguments.

## Annotation preparation utilities

### Convert LabelMe polygons to YOLO segmentation labels

```powershell
python scripts/labelme_to_yolo_seg.py
```

### Generate mask-based crops from LabelMe annotations

```powershell
python scripts/labelme_to_mask.py
```

## Trait analysis and clustering

The repository includes several downstream analysis scripts for trait filtering, clustering, and visualization:

- [scripts/trait_selection_hierarchical_clustering.py](scripts/trait_selection_hierarchical_clustering.py)
- [scripts/dus_type_clustering_pipeline.py](scripts/dus_type_clustering_pipeline.py)
- [scripts/core_trait_pca_kmeans_pipeline.py](scripts/core_trait_pca_kmeans_pipeline.py)
- [scripts/analyze_table3.py](scripts/analyze_table3.py)

These scripts cover:

- missingness-based trait filtering
- imputation and normalization
- ANOVA and random-forest-based trait ranking
- PCA, UMAP, hierarchical clustering, and KMeans
- cluster visualization and trait heatmaps

## Variety-level core fingerprint workflow

The variety-level fingerprint analysis is centered on:

- [scripts/build_variety_core_fingerprint.py](scripts/build_variety_core_fingerprint.py)
- [scripts/compare_core_fingerprint_schemes.py](scripts/compare_core_fingerprint_schemes.py)
- [scripts/internal_validate_core_fingerprint.py](scripts/internal_validate_core_fingerprint.py)
- [scripts/rebuild_internal_validation_s7.py](scripts/rebuild_internal_validation_s7.py)
- [scripts/refresh_core_scoring_from_s3_s5.py](scripts/refresh_core_scoring_from_s3_s5.py)
- [scripts/build_supplementary_table_s6.py](scripts/build_supplementary_table_s6.py)

This part of the workflow covers:

- encoded fingerprint-matrix generation
- core-trait scoring
- exhaustive subset search
- forward-selection trace analysis
- objective-scheme comparison
- audited bootstrap OOB and constrained reference-evaluation assessment
- frozen-code measurement-error sensitivity analysis
- fully nested selection-stability analysis

Representative outputs are stored in:

- [outputs/variety_core_fingerprint](outputs/variety_core_fingerprint)
To inspect the audited reference-evaluation and nested-selection method:

```powershell
python scripts/rebuild_internal_validation_s7.py --help
```

The numerical rebuild requires an unencoded trait matrix, paired expert/image-derived agreement inputs, and user-supplied manuscript/supplementary templates. Generated statistics and revised documents are written to the user-selected output directory and are not included by this methods-only update.

## Final paper-ready model results

The final selected retrained runs currently included in this repository are:

- Whole-plant organ segmentation:
  - [results/paper_seg/maize_plant_paper_aug_b2/paper_metrics.json](results/paper_seg/maize_plant_paper_aug_b2/paper_metrics.json)
- Kernel instance segmentation:
  - [results/paper_seg/maize_kernel_paper_v3/paper_metrics.json](results/paper_seg/maize_kernel_paper_v3/paper_metrics.json)

A compact summary table is provided in:

- [docs/model_validation_summary.csv](docs/model_validation_summary.csv)

## Data availability

This repository does not bundle the full raw image datasets by default. Instead, it is intended to provide:

- code
- configuration files
- selected output tables
- selected model-evaluation summaries
- optional small demo subsets under `data/demo/`

See [data/README.md](data/README.md) for the intended demo-data layout.

## Notes on curation

This repository was assembled from a larger active research workspace. As a result:

- only selected manuscript-relevant scripts are included here;
- filenames have been partially normalized for public readability;
- manuscript-facing scripts use repository-relative defaults or explicit command-line inputs; historical exploratory scripts may still require user-supplied data paths.

The public release is intended to prioritize manuscript support, methodological transparency, and computational reproducibility.

## Citation

If you use this repository or its workflow in your work, please cite the associated manuscript and include the repository URL once publicly available.
