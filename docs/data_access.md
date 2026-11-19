# Data availability

Small image examples, best.pt/last.pt checkpoints and the ONNX deployment model are included in this repository.
Complete image, annotation and training packages are distributed through the
[v2.1.0 Release](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/tag/v2.1.0); see [downloads](../data/README.md).

The Release contains 958 indoor images, 150 outdoor images and a separate 256-image annotated
model-development collection. The model-development collection and the 150-variety phenotype panel
contain no overlapping varieties. PyTorch evaluation of best.pt and ONNX deployment used the same
model weights, as [confirmed by the authors](author_confirmation_2026-11-19.md) on 19 November 2026.
The current Release README and dated confirmation supersede earlier model-identity and overlap
descriptions in the tagged source snapshot and original ZIPs. Known content-to-prior-source filename
correspondences are retained with the complete data and do not establish cultivar equivalence.

Complete historical phenotype/reference-evaluation inputs and the original C++ measurement application
are not included. The release therefore does not reproduce the full historical raw-image-to-trait pipeline.
Images/annotations are CC BY-NC 4.0; code and checkpoints retain applicable upstream terms.
