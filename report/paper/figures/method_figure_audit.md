# Method-figure audit

The two original schematics use soft panel grouping and serif labels, following the supplied Group report. All panels, arrows, text and feature/prompt icons are vector objects. The exact reference, query, reference mask, and predictions are from archived COCO episode `coco-f3-q000061-c0bcbff23c1abff2-r000` (case06). Actual CG-ICS candidates and selected concept `red octagon` are recorded in `assets/example_provenance.json`. Feature grids, cluster-color grids and the query-box icon are schematic, not claimed intermediate outputs. No query ground truth is an inference input.

INSID3 keeps original features for query grouping and seed affinity, with positional debiasing before reference correspondence. The reference mask bypasses the image encoder. Reference correspondence guides seed selection and aggregation; the final threshold is retained. CG-ICS separates reference-driven concept generation/scoring/search from reference-query visual transfer and combines the resulting concept/boxes in a final query prediction. Arrows express logical dependencies, not concurrent execution.

All archive inputs are local and bundled as exact small assets for standalone regeneration. Primary-paper and pinned-source provenance is retained in the JSON audit.
