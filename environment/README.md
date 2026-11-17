# Software versions

The following versions were read from the local environment used for the non-training checks. They describe the current installation, not the original model-training environment.

| Software | Observed version |
| --- | --- |
| Python | 3.13.5 |
| ultralytics | 8.4.46 |
| torch | 2.11.0+cu126 |
| torchvision | 0.26.0+cu126 |
| opencv-python | 4.12.0.88 |
| numpy | 2.4.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.3 |
| openpyxl | 3.1.5 |
| matplotlib | 3.10.0 |
| python-docx | 1.2.0 |
| ONNX Runtime | 1.23.2, used from the separately installed local inference packages |

The statistical dependency pins are retained in [requirements.txt](../requirements.txt). This installation record is not a newly tested clean-install lockfile.

Model-specific records:

- Five-class ONNX export: Ultralytics 8.3.32 and PyTorch producer 2.5.1.
- Later six-class checkpoint: saved Ultralytics 8.3.89; original training PyTorch version [Pending verification].
- Current inference checks: Ultralytics 8.4.46 and PyTorch 2.11.0+cu126; CPU inference. ONNX checks additionally use ONNX Runtime 1.23.2.

These are different records and must not be collapsed into one purported historical environment. The C++ image-measurement application is separate from the public Python code. See [model validation notes](../docs/yolo11_validation.md).
