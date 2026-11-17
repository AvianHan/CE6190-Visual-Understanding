# Figure 1–2 polish, 2026-11-17

The two diagrams were rebuilt as compact, independently readable method figures after inspecting the submitted Group report Figure 1. They preserve the established five-color palette, exact archived example inputs and final masks, and the scientific pathways audited against the primary papers and pinned source.

## Visual changes

- Equal 6.875 × 3.0 inch canvases with aligned, lightly tinted panels, thin orthogonal arrows, white modules, and consistent STIX serif type.
- All labels are 7.1–8.5 pt at the intended figure width, below report body size; panel headings no longer dominate the page.
- INSID3 follows input/feature extraction → correspondence → region selection. A separate blue path explicitly carries the original query features into seed affinity.
- CG-ICS separates reference-driven concept guidance, stitched visual transfer, and the final prediction. The query-to-visual branch is disconnected from the MLLM input path. Search is a controller loop, not an extra operation claimed to be inside SAM 3.
- The final predictions are the actual archived outputs of case06. Feature/cluster grids and the query-box icon are schematic and are not represented as measured intermediate activations.

## Scientific checks

- The reference mask bypasses the INSID3 image encoder. Original query features serve clustering and seed affinity; positional debiasing precedes cross-image correspondence.
- INSID3 retains the executed cross-image similarity × seed affinity × candidate coverage score and the final threshold.
- CG-ICS concept generation receives reference views. Its SAM 3 candidate score uses reference IoU × query presence; query ground truth is absent.
- CG-ICS converts reference mask components to spatial boxes on the stitched image, obtains a query-side mask, converts that mask to query boxes, and combines those boxes with the selected concept for final SAM 3 prediction.
- The selected concept “red octagon” and the illustrative alternative candidate names are grounded in the archived case06 record.

## Validation

Both final PNG renders were visually inspected. A separate scientific review of the final renders passed, with no incorrect dependency, query-GT input, or fatal arrow issue. Text bounds are asserted against the canvas by the generator; both figures report no out-of-canvas text. The generator rejects fonts exceeding 8.5 pt. PDF and SVG preserve text, panels, arrows and schematic icons as vector elements, with only real example image/mask thumbnails embedded as rasters. Source/provenance hashes and output hashes are in `paper/figures/method_figure_audit.json`.

Reproduce from the report root:

```bash
python3 paper/build_method_figures.py
```

No experiment, metric, result record, or caption in the report source was changed by this figure revision.
