# Maize phenotyping and phenotypic fingerprinting

Code, whole-plant segmentation weights and small image examples for the maize phenotyping study.
Complete datasets and training evidence are provided as [versioned Release downloads](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/tag/v2.1.0).

| Directory | Contents |
| --- | --- |
| [models](models/) | New best/last PT weights and the retained historical ONNX |
| [data](data/) | Indoor/outdoor examples, annotated examples and full-data download links |
| [scripts](scripts/) and [analysis](analysis/) | Statistical analysis and segmentation commands |
| [configs](configs/) and [data_schema](data_schema/) | Analysis settings, class IDs and input templates |
| [environment](environment/) | Software versions |
| [docs](docs/) | Methods and validation scope |

## Run

```powershell
python -m pip install -r requirements.txt
# Install the segmentation dependencies listed in environment/README.md.
python scripts/predict_plant_seg.py --model models/best.pt --input data/examples/outdoor/whole_plants --output outputs/segmentation --imgsz 1024
```

For phenotype analysis, see [scripts/README.md](scripts/README.md). Original phenotype inputs are required.

## Model and data

The new YOLO11s-seg uses five classes and a fixed 256-image train/validation/test split of 154/51/51.
Test mask AP50: 0.796822; mask AP50:95: 0.450467; foreground class-union mIoU: 0.673647.
See [validation details](docs/model_notes.md), including the quarter-resolution IoU grid and unverified
overlap with the original 150-variety panel. The historical ONNX is a separate artifact.
Existing phenotype values, coding and selection results were not recalculated.

The complete Release contains 958 indoor examples, 150 outdoor examples and the separate annotated
training collection. See [data/README.md](data/README.md) for downloads and [availability scope](docs/data_access.md).

## License and citation

Code: [AGPL-3.0](LICENSE). Images and annotations: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Checkpoints retain applicable upstream terms. Cite the manuscript and the version in [CITATION.cff](CITATION.cff).
