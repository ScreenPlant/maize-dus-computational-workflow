# Author confirmation, 19 November 2026

The authors confirmed on 19 November 2026 that the 256-image model-development collection and the
150-variety phenotype panel contain no overlapping varieties, and that the same model weights were
used for PyTorch evaluation and ONNX deployment. This supersedes the earlier unverified overlap
status recorded before that confirmation.

These are author confirmations. A cross-identifier mapping and an export-equivalence test were not
independently evaluated in this documentation update. The actual checkpoint/ONNX hashes, export
metadata and frozen 154/51/51 split remain unchanged. Distinct file formats do not have identical
file hashes. The retained ONNX has a 1920-pixel input and its recorded export metadata remain
Ultralytics 8.3.32 / PyTorch producer 2.5.1; the new PT training/test environment is Ultralytics 8.4.46.
The reported mask AP and quarter-grid class-union IoU are measurements of best.pt under the
1024-pixel evaluation protocol, not a separate ONNX performance test.

No phenotype values, coding rules or fingerprint-selection results were recomputed. Model-data
separation does not make the 1,000 repeated 120/30 splits of the phenotype panel external validation.
The full historical repeated-measurement image archive is not supplied by the curated examples.

The v2.1.0 ZIP assets are retained without replacement. Their earlier provenance/status descriptions
should be read with this dated addendum. The tagged Release is public, but no DOI was obtained.
