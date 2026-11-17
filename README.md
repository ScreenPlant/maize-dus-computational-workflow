# Reliability- and robustness-constrained maize research fingerprints

This revision implements a research phenotypic fingerprint workflow for a single-site/year
maize panel. It does not establish formal DUS distinctness, external validity, or reliable
reference-to-image identity matching. The model-based robustness target concerns uniqueness,
and complete reference-image code agreement is a separate outcome.

## Current entry points

1. `scripts/build_variety_table.py`: aggregate private 20-object paired records and 3-image x
   3-expert ratings to one reference/image value per variety and trait. T28 is a ratio of means.
2. `scripts/run_revised_analysis.py`: reliability gates, exhaustive search, 10,000 all-subset
   confirmation simulations, independent 10,000-simulation selected-subset auditing, and
   1,000 complete endosperm-stratified 120/30 reselections. Evaluation accessions never fit
   reliability statistics, coding parameters, discrepancy calibration, subset choice or audits.
3. `scripts/summarize_nested.py`: retain all requested runs, including failures, in event denominators.
4. `scripts/predict_plant_seg.py` and `scripts/evaluate_plant_seg.py`: artifact-specific segmentation
   demonstration/evaluation, separate from statistical fingerprint refitting.
5. `scripts/trait_agreement.py`: paired variety-level agreement, object variability and expert-rating evidence.
6. `scripts/audit_gate_sensitivity.py` and `scripts/audit_model_sensitivity.py`: stricter reliability/exact-agreement gates and specified alternative discrepancy models. Sensitivity results do not replace the primary selection.

Numerical/count gates are CCC >=0.80 and MAPE <10%; ordered-state linear kappa and nominal-state
unweighted kappa must be >=0.60. Complete paired coverage and dominant-code fraction <0.90
are required; deterministic ratio T28 is excluded as an independent candidate. Acceptance requires
a 95% Wilson lower bound >=0.95 for complete simulated uniqueness. Size ranking follows acceptance;
failed independent audits are retained without repeated selection. See `configs/analysis_protocol.json`.

## Run with authorized private inputs (PowerShell)

```powershell
python -m pip install -r requirements.txt
python scripts/build_variety_table.py --raw D:/authorized/object_ratings.xlsx --materials D:/authorized/materials.csv --output D:/authorized/rebuilt
python scripts/run_revised_analysis.py --inputs D:/authorized/rebuilt --membership D:/authorized/membership1000.csv --output D:/analysis/revised --scope all --workers 4
python scripts/summarize_nested.py --results D:/analysis/revised --output D:/analysis/summary --runs 1000
```

Full authorized inputs are required; templates are under `data_schema/`. The reader should not
expect examples to reproduce the complete panel results. Source code is Python. The historical
C++ measurement application is not included. See `docs/measurement_scope.md`, `docs/data_access.md`
and `environment/README.md`. Trained whole-plant ONNX parameters and their manifest are under `models/`.

## Availability status

The source, protocol and deployment model are included in this revision. Images and annotations
are outside the current public release. Versioned archives are listed under the repository's
[Releases](https://github.com/ScreenPlant/maize-dus-computational-workflow/releases).
A software-archive DOI has not yet been assigned. No complete indoor imaging dataset or full
protected phenotype input is represented as publicly deposited. Earlier default five/six-trait
and 100-split scripts are retired in this tree; their paths contain notices, and Git history is retained.
