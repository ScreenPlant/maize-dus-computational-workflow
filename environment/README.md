# Environments

The pinned root requirements describe the statistical reanalysis environment (Python 3.13.5).
Model-development and C++ application environments are separate. The ONNX metadata record
Ultralytics 8.3.32 and PyTorch producer 2.5.1; the latter is an export-producer version,
not a verified historical training environment. Retained six-class training records identify
Ultralytics 8.3.89. Do not install a single version and claim it reconstructs every artifact.
The actual C++ project used MSVC v143, MFC, OpenCV 4.10.0 link libraries and ONNX Runtime;
its source is outside this public release. The exact C++ runtime library versions remain documented
only where verified. Additional evaluator environments must be recorded with their outputs.
