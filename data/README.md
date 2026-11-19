# Image examples and full datasets

`examples/indoor/` contains one image for each of seven organs; `examples/outdoor/whole_plants/`
contains one whole-plant image. These are curated examples without newly paired segmentation annotations.
`examples/annotated_whole_plants/` contains two images and YOLO polygon labels from the fixed test split.
Class IDs are listed in [plant_deployment_classes.json](../configs/plant_deployment_classes.json).

The annotated example basenames are shortened for portability; their image/label bytes are unchanged.
Use [examples_manifest.csv](examples_manifest.csv) to identify their original entries by SHA-256 in
the complete training archive. They remain test examples and must not be added to a training set.

## Complete downloads

Complete data are [Release attachments](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/tag/v2.1.0), separate from the small examples above.

| File | Contents |
| --- | --- |
| [whole_plant_training_archive.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/whole_plant_training_archive.zip) | 256 annotated images, fixed split, weights and evaluation |
| [indoor_seedlings.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_seedlings.zip) | 150 seedling images |
| [indoor_leaves.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_leaves.zip) | 150 leaf files |
| [indoor_ears.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_ears.zip) | 150 ear images |
| [indoor_cobs.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_cobs.zip) | 150 cob images |
| [indoor_kernels.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_kernels.zip) | 150 kernel images |
| [indoor_anthers.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_anthers.zip) | 150 anther images |
| [indoor_silks.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/indoor_silks.zip) | 58 silk images |
| [outdoor_whole_plants.zip](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases/download/v2.1.0/outdoor_whole_plants.zip) | 150 whole-plant examples |

The Release includes README.md, LICENSE_DATA.txt, SHA256SUMS.txt, image_file_manifest.csv and
image_content_source_correspondence.csv. Extract the eight example-image ZIPs into one common
directory; extract the training archive separately and follow its README.

Images and annotations are CC BY-NC 4.0. The curated collection is not the complete historical
repeated-measurement archive. Filename IDs are depositor-supplied; checksums do not authenticate
cultivar identity. Known prior source-filename correspondences and their limits are recorded with
the complete data. See [availability scope](../docs/data_access.md).
