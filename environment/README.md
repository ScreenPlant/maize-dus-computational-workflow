# Software versions

This folder records software versions. Install the statistical dependencies with the root [requirements.txt](../requirements.txt).

| Component | Version information |
| --- | --- |
| Statistical analysis | Python 3.13.5; package versions in `requirements.txt` |
| Released plant ONNX file | Ultralytics 8.3.32 and PyTorch producer 2.5.1 recorded in the export metadata |
| Current inference check | Ultralytics 8.4.46, PyTorch 2.11.0+cu126 and ONNX Runtime 1.23.2; CPU execution |
| Image-measurement application | C++, MSVC v143/MFC and OpenCV 4.10.0 link libraries, with ONNX Runtime |

The export metadata and the current inference versions describe different steps. The original image-measurement application is separate from the public Python analysis scripts.

The retained six-class training checkpoint records Ultralytics 8.3.89. It is distinct from the released five-class model; see [model notes](../docs/model_notes.md).

The current inference environment is also recorded in the [v2.0.0 release](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/tag/v2.0.0). It describes the tested installation, rather than the original training environment.
