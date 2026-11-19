# Software versions

| Stage | Versions |
| --- | --- |
| Statistical analysis | Python 3.13.5; packages in ../requirements.txt |
| New PT retraining/test | Python 3.13.5; Ultralytics 8.4.46; torch 2.11.0+cu126; torchvision 0.26.0+cu126; CUDA 12.6; RTX 3060 |
| New PT image/analysis dependencies | OpenCV-Python 4.12.0.88; NumPy 2.4.3; SciPy 1.15.3; pandas 2.2.3 |
| Source annotations | LabelMe 5.5.0 |
| Historical ONNX export metadata | Ultralytics 8.3.32; PyTorch producer 2.5.1 |
| ONNX inference check | Ultralytics 8.4.46; torch 2.11.0+cu126; ONNX Runtime 1.23.2; CPU |
| Historical C++ application | MSVC v143/MFC; OpenCV 4.10.0 link libraries; ONNX Runtime; source not included |

Install statistical dependencies with `python -m pip install -r requirements.txt`.
For the new PT, install the matching PyTorch/torchvision build and Ultralytics 8.4.46.
For ONNX inference, also install ONNX Runtime. Full environment and evaluator instructions are in the training archive.
