# Input tables

This folder describes the input format used by the analysis. The CSV files are empty templates and contain no cultivar records.

| File | Contents |
| --- | --- |
| `materials_template.csv` | Variety ID, name, endosperm type and breeding type |
| `variety_summary_template.csv` | Reference and image-derived values for each variety and trait |
| `membership_template.csv` | Reference/evaluation membership for each repetition |
| `trait_metadata.json` | Trait names, types, units and GB/UPOV correspondence |

`variety_summary.csv` has four columns: `Variety`, `Trait`, `Reference` and `Image`. Include each variety–trait row and leave unavailable measurements blank. Numerical values use the units in `trait_metadata.json`.

The membership table identifies 120 reference and 30 evaluation varieties in each repetition. `Role` is either `Reference` or `Evaluation`. The same membership file is needed to reproduce the study's 1,000 repetitions.

Keep local files in `private_inputs/`, which is excluded from Git tracking. See [scripts/README.md](../scripts/README.md) for commands.

## Image and training-data manifests

The following CSV files contain headers only. No sample identities or invented observations are supplied.

| Template | Fields |
| --- | --- |
| `yolo_dataset_manifest_template.csv` | `image_id`, `variety_id`, `plant_id`, `annotation_id`, `split`, `source_dataset`, `overlap_with_150_panel` |
| `trait_image_manifest_template.csv` | `image_id`, `variety_id`, `trait_id`, `organ`, `acquisition_environment`, `image_filename`, `data_availability` |

In the YOLO manifest, image and annotation IDs link actual paired files; variety and plant IDs identify their biological groups. `split` records the actual train/val/test assignment, or `unknown`. `source_dataset` records provenance. `overlap_with_150_panel` is `yes`, `no` or `unknown`, supported by identity records rather than filename similarity alone.

In the trait-image manifest, `organ` identifies the sampled organ; `image_filename` identifies the actual image. `acquisition_environment` is `indoor` or `outdoor`. `data_availability` records `public`, `restricted` or `pending_upload` as appropriate.

The study acquisition scope is:

- Outdoor whole-plant images with YOLO11-seg: T26 and T27.
- Derived, without separate imaging: T28 = mean(T26) / mean(T27) for the same variety and measurement source. Do not create an image record for T28.
- Indoor images with conventional processing: T1, T2, T3, T6, T7, T9, T10, T23, T24, T30, T31, T32, T33, T34, T36, T37, T40, T41, T42 and T43.

The full manifests remain pending. Public identifiers and files must be cleared for release before deposition.
