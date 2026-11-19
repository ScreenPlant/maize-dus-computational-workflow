# Image measurement scope

The historical application combined whole-plant model localization with C++ geometric and
organ-specific conventional processing. That C++ source is not included in this release.
T6/T7 have a model-assisted leaf-skeleton/angle path; indoor acquisition does not identify
the computational method. T26 uses an ear lower-boundary proxy, with a silk-box fallback;
T27 uses the tassel-box top. T28 in the analysis is the ratio of component variety means.
T31 uses the maximum of sampled transverse image widths. The conventional T36/T37 path
constructs a size-estimation rectangle from neighboring kernel-center spacings. T32 estimates grain-row number from a single lateral ear image by doubling the visible-row estimate.
This assumes that the visible rows approximate half the circumference; it is not a direct observation of
all circumferential rows. Occlusion, perspective and irregular row arrangement can bias this approximation,
and doubling cannot reproduce odd-valued reference row counts exactly. No paired cross-sectional
validation was added, and the original trait values were not recalculated.
Ear processing did not enable the optional AI module in 2025. Colour templates are BGR/HSV
rules rather than CIEDE2000 calibration; residual ΔE00 was not measured.

The public scripts start from the documented phenotype-record schema for statistical analysis,
and separately demonstrate segmentation of supplied images. They do not implement an end-to-end
raw-image-to-23-trait replacement for the historical C++ application.
