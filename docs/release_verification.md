# Release migration checks

The public entry points were checked locally using authorized protected inputs, without
placing those inputs in this repository.

- Object/rater aggregation reproduced both frozen variety-level input arrays exactly.
- The scientific engine functions and selection function retained their original abstract syntax;
  the input loader and descriptive metadata were adapted for portable CSV inputs.
- Recomputing the complete 150-variety search reproduced 1,023 enumerated subsets, 57 unique
  in both observed sources and six confirmation-qualified subsets. The representative minimum
  qualified configuration had nine traits. Its independent audit reproduced 9,712/10,000
  complete-uniqueness outcomes.
- One complete nested refit through the new Windows process-pool entry point reproduced its
  archived selection, independent audit and held-out evaluation exactly.
- The summary entry point reproduced archived all-1,000-run event counts: 934 qualified,
  929 qualified with unique evaluation codes, and 802 qualified with no mixed-library collisions.
  This migration check did not rerun all 1,000 archived refits.
- All 45 columns of the 23-trait agreement/descriptor table matched the archived table.
- The ONNX artifact loaded and produced finite outputs at 1 x 3 x 1920 x 1920 in ONNX Runtime
  1.23.2 (CPU). A supplied-image inference smoke check also completed. These are software
  execution checks, not segmentation AP/IoU or evidence of biological generalization.

Exact study-level replication still requires the protected inputs and frozen membership file.
Public example images are not included in this release.
