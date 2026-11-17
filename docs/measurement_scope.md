# Image measurements

The 23-trait study scope is as follows:

| Traits | Acquisition | Processing |
| --- | --- | --- |
| T26, T27 | Outdoor whole-plant images | YOLO11-seg localization followed by geometric measurement |
| T28 | No separate imaging | Ratio of the corresponding variety means: mean(T26) / mean(T27) |
| T1, T2, T3, T6, T7, T9, T10, T23, T24, T30, T31, T32, T33, T34, T36, T37, T40, T41, T42, T43 | Indoor organ images | Conventional image processing; no additional trained segmentation models |

This scope follows the authors' confirmed study workflow. In particular, T6 and T7 are indoor conventional-image traits. Additional model-assisted routines present in retained source are not described as their reported processing path.

The retained whole-plant geometry routine uses the upper-ear region for T26, with a silk-region fallback, and the tassel top relative to the ground reference for T27. The exact run-specific settings should be retained with the image-processing records.

Conventional measurements include sampled transverse ear widths (T31), a doubled visible-row estimate (T32), and kernel-size estimates based on neighboring center spacings (T36/T37). The source-based T32 procedure and its assumptions are described in [yolo11_validation.md](yolo11_validation.md).

Indoor colour processing uses reference-based white balancing and organ-specific BGR/HSV rules; residual ΔE00 was not measured. The public code provides statistical analysis and model inference. The original C++ raw-image measurement application is outside the release scope.
