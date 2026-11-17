# Plant model and training records

The released five-class ONNX model segments leaf, root, ear, silk and tassel regions. Its export metadata record Ultralytics 8.3.32, PyTorch producer 2.5.1 and an input size of 1920 × 1920.

The retained six-class checkpoint is a different record. It lists Ultralytics 8.3.89 and nominal training/validation/test partitions of 164/40/40 images, with 2,831/729/722 polygons. Its nominal-test mask AP50 and AP50:95 are 0.734 and 0.407. These values apply to that checkpoint, not to the released five-class model, and they are not pixel mean IoU.

The original five-class training manifest, split membership and overlap with the 150-variety phenotype panel have not been established from the retained records. Variety-level nested fingerprint evaluation does not provide an independent test of the segmentation model.

The released ONNX model has been checked for loading and single-image inference. Those software checks are separate from segmentation AP/IoU evaluation.
