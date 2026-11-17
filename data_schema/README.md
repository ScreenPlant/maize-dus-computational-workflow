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
