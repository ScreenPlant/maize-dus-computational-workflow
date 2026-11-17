# Maize phenotyping and phenotypic fingerprinting

Code and a trained whole-plant segmentation model for our maize phenotyping study. The analysis includes trait agreement, reliability- and robustness-constrained trait selection, and internal reference–evaluation analysis.

## Contents

| Directory | Contents |
| --- | --- |
| [scripts](scripts/) | Trait analysis, fingerprint selection and plant segmentation |
| [analysis](analysis/) | Functions used by the analysis scripts |
| [models](models/) | Trained whole-plant ONNX model and class definitions |
| [configs](configs/) | Analysis settings and model classes |
| [data_schema](data_schema/) | Input table templates and trait definitions |
| [environment](environment/) | Software versions |
| [docs](docs/) | Method and data availability notes |

## Installation

The statistical analysis uses Python 3.13.5.

```powershell
python -m pip install -r requirements.txt
```

Plant segmentation additionally requires Ultralytics and ONNX Runtime. The versions used for the inference check are listed in [environment/README.md](environment/README.md).

## Fingerprint analysis

Prepare `materials.csv`, `variety_summary.csv` and the reference–evaluation membership file using the [input templates](data_schema/README.md). Place local inputs in `private_inputs/`.

```powershell
python scripts/run_revised_analysis.py --inputs private_inputs --membership private_inputs/membership.csv --output outputs --scope all
python scripts/summarize_nested.py --results outputs --output outputs/summary --runs 1000
```

Traits are screened for reliability before subset selection and model-based robustness evaluation. Internal assessment uses 1,000 complete nested reselections. See the [method description](docs/methods.md) and [script list](scripts/README.md) for details.

## Plant segmentation

```powershell
python scripts/predict_plant_seg.py --model models/plant_deployment_5class.onnx --input private_inputs/images --output outputs/segmentation
```

The model segments leaf, root, ear, silk and tassel regions. See [models/README.md](models/README.md) for the model file and class order.

## Data availability

This release provides code and the trained plant model. Cultivar-linked phenotype records, images and annotations are not publicly released because of confidentiality restrictions. Input templates contain column definitions only. Reproducing the study results requires the original study inputs and membership file.

The fingerprint analyses describe the 150-variety, single-location and single-year study panel. Image-measurement methods and the scope of the public code are described in [docs/measurement_scope.md](docs/measurement_scope.md).

## Citation

Please cite the associated manuscript and the software version used. Citation information is provided in [CITATION.cff](CITATION.cff).
