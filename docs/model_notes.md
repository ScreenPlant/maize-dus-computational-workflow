# Whole-plant training and validation

YOLO11s-seg was retrained on 256 images (one variety per image, author-confirmed), annotated in
LabelMe 5.5.0 with 4,225 polygons: leaf 3,084; root 152; ears 290; silk 335; tassel 364.
The fixed variety-grouped train/validation/test split contains 154/51/51 images and 2,539/848/838
polygons. Allocation used seed 42 and annotation-count balancing, without model performance.
Overlap with the 150-variety phenotype panel is unverified because a confirmed identifier crosswalk
is unavailable. This held-out split is internal to the annotated collection, not an external cohort.

Training requested 200 epochs and completed 155. best.pt is epoch 105, selected by validation
box AP50:95 plus mask AP50:95; last.pt is retained separately. No test-based selection was used.

| Test metric (51 images, 838 instances) | Estimate | 95% image-bootstrap interval |
| --- | --- | --- |
| Mask AP50 | 0.796822 | 0.753157–0.851075 |
| Mask AP50:95 | 0.450467 | 0.423200–0.495946 |
| Foreground class-union mIoU | 0.673647 | 0.636509–0.707232 |

AP settings: confidence 0.001, NMS IoU 0.7, max_det 300, FP32, maximum input side 1024, no augmentation.
Union IoU uses confidence >=0.25 and quarter-resolution letterboxed masks (mask_ratio=4), not native
resolution or matched-instance IoU. Ground-truth and predicted instances are unioned per class;
intersections/unions are accumulated over test images, then averaged over five foreground classes,
excluding background. False-positive regions are included. Intervals use 1,000 image bootstrap samples,
seed 20261118. The initial statistics export failed after inference; an identical frozen-model/settings
rerun exported results without test tuning. Logs and full metrics are included in the training archive.

The new PT was retrained after loss of the historical PT. The retained ONNX (input 1920x1920;
export metadata Ultralytics 8.3.32 / PyTorch producer 2.5.1) is a separate artifact. New PT metrics
must not be attributed to that ONNX. Original phenotype values, coding and selection results are unchanged.

Download [whole_plant_training_archive.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/whole_plant_training_archive.zip) for original
LabelMe/YOLO annotations, frozen membership, environment, run settings, test results and reproduction
commands. Original scripts retain historical paths; a portable evaluator is also provided.
