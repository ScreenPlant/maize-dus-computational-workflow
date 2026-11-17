# Maize DUS phenotyping and phenotypic fingerprinting

Code and model files supporting the maize phenotyping study. The analysis covers 23 traits, reliability- and robustness-constrained fingerprint selection, and 1,000 nested reference–evaluation repetitions.

YOLO11-seg is used only for outdoor whole-plant images supporting T26 and T27. T28 is the ratio of their variety means. The other 20 traits use conventional processing of indoor images.

## Repository structure

| Directory | Contents |
| --- | --- |
| `scripts/`, `analysis/` | Analysis and inference scripts; supporting functions |
| `models/` | Five-class deployment ONNX and a separately identified later six-class checkpoint |
| `configs/` | Analysis settings and deployment classes |
| `data_schema/` | Input formats and empty image/dataset manifest templates |
| `environment/` | Recorded software versions |
| `docs/` | Methods, model validation and data availability |

## Installation

Statistical analysis uses Python 3.13.5:

```powershell
python -m pip install -r requirements.txt
```

Model inference additionally uses Ultralytics and ONNX Runtime. The tested versions and their scope are listed in [environment/README.md](environment/README.md).

## Running the code

Prepare the study input tables following [data_schema/README.md](data_schema/README.md), then run:

```powershell
python scripts/run_revised_analysis.py --inputs private_inputs --membership private_inputs/membership.csv --output outputs --scope all
python scripts/summarize_nested.py --results outputs --output outputs/summary --runs 1000
```

For the existing five-class deployment model:

```powershell
python scripts/predict_plant_seg.py --model models/plant_deployment_5class.onnx --input private_inputs/images --output outputs/segmentation --imgsz 1920
```

Commands require the relevant local inputs; the repository does not include the original study data.

## Data and models

Model downloads, classes and the later checkpoint's scope are described in [models/README.md](models/README.md). See [YOLO11 validation](docs/yolo11_validation.md) and [image-measurement methods](docs/measurement_scope.md).

Training images, annotations and indoor images are pending separate data deposition. Links will be added after release. Current availability is detailed in [docs/data_availability.md](docs/data_availability.md); complete raw-data reproduction is not yet supported.

## Citation and version

The existing [v2.0.0 release](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/tag/v2.0.0) and [CITATION.cff](CITATION.cff) are retained. This branch prepares reproducibility updates for a possible v2.1.0 release. No new release or DOI has been created.
