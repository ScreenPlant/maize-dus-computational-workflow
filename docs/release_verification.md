# Numerical checks

The public scripts were checked using the retained study inputs.

- Aggregation reproduced the reference and image-derived variety summaries.
- The full-panel search reproduced 1,023 subsets: 57 unique in both observed sources and six meeting the simulation criterion. The smallest qualifying configurations contained nine traits.
- The selected configuration reproduced 9,712 complete-uniqueness outcomes in 10,000 independent-seed simulations.
- A nested refit reproduced its saved selection and evaluation. Summarizing the existing 1,000 runs reproduced 934 qualified selections, 929 qualified selections with unique evaluation codes, and 802 qualified selections without mixed-library collisions.
- The complete 23-trait agreement and descriptor table matched the retained table.

The ONNX model also loaded and completed single-image inference. This execution check is separate from segmentation accuracy evaluation.
