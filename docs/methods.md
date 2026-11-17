# Fingerprint analysis

## Data and trait screening

Numerical traits use the means of 20 paired reference and image observations per variety. Categorical references use three images scored by three experts; program categories use three outputs. The derived ratio uses the component variety means. Trait definitions and units are listed in `data_schema/trait_metadata.json`.

Traits are screened before subset selection. Numerical/count traits require CCC ≥ 0.80 and MAPE < 10%; ordered traits require linear κ ≥ 0.60, and nominal traits require unweighted κ ≥ 0.60. Complete paired coverage and a dominant-code fraction < 0.90 are required. The deterministic ratio T28 is excluded as an independent candidate.

## Subset selection

All non-empty eligible subsets are examined. A qualifying subset must be unique in both observed sources and have a 95% Wilson lower bound ≥ 0.95 for complete uniqueness in 10,000 calibrated simulations. The minimum trait number is considered among qualifying subsets.

The selected subset is evaluated in another 10,000 simulations using an independent seed. A failed result is retained without repeated selection. The discrepancy model uses paired reference–image differences, rather than independent repeat-imaging measurements; its reference values and coding parameters remain fixed during simulation.

## Internal evaluation

The study uses 1,000 endosperm-stratified 120/30 reference–evaluation allocations. Trait screening, coding, discrepancy calibration and selection are repeated within each reference set. Evaluation varieties are assessed after selection.

Selection success, complete uniqueness and strict transfer are summarized across runs, including unsuccessful selections. Complete reference–image code agreement is measured per variety. Pairwise collision rates use comparison-specific denominators; repeated comparisons are not treated as independent biological samples.

These analyses assess research phenotypic fingerprints within the single-location, single-year panel. Code uniqueness is separate from reference–image code agreement and does not establish formal DUS distinctness or generalization across environments.
