# Scripts

| Script | Purpose |
| --- | --- |
| `build_variety_table.py` | Summarize object measurements and expert scores by variety |
| `trait_agreement.py` | Calculate agreement, within-variety variation and expert agreement |
| `run_revised_analysis.py` | Select traits with reliability and robustness criteria; run nested evaluation |
| `summarize_nested.py` | Summarize selection outcomes, code agreement and pairwise collisions |
| `audit_gate_sensitivity.py` | Compare stricter reliability and agreement criteria |
| `audit_model_sensitivity.py` | Compare discrepancy models and trait inclusion |
| `predict_plant_seg.py` | Segment whole-plant images using the released ONNX model |
| `evaluate_plant_seg.py` | Evaluate a specified segmentation model and annotated dataset |

Run a script with `--help` for its input arguments. Earlier script names contain notices directing users to the current commands.

## Prepare variety summaries

```powershell
python scripts/build_variety_table.py --raw private_inputs/object_ratings.xlsx --materials private_inputs/materials.csv --output private_inputs/summary
```

The script summarizes 20 paired numerical measurements and nine expert scores / three program outputs for categorical traits. The derived ratio uses the component variety means.

## Run the analysis

```powershell
python scripts/run_revised_analysis.py --inputs private_inputs/summary --membership private_inputs/membership.csv --output outputs --scope all
python scripts/summarize_nested.py --results outputs --output outputs/summary --runs 1000
```

The study uses 1,000 reference–evaluation allocations. Their membership file is required to reproduce the reported analysis. Input formats are described in [data_schema/README.md](../data_schema/README.md).
