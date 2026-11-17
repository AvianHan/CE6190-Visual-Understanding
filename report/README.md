# CE6190 Individual Project report

**Visual Correspondence and Concept Guidance for In-Context Segmentation**  
HAN SHUANGYUE · G2690005A · [hans0053@e.ntu.edu.sg](mailto:hans0053@e.ntu.edu.sg)

Repository: <https://github.com/AvianHan/CE6190-Visual-Understanding>

This directory contains the editable report, bibliography, four figures, source
images for those figures, citation/logic reviews, and the exported report PDF.
The report's empirical conclusions use audited first-batch results. The proposed
AGS extension is presented with its derivations and evaluation design; second-batch
outcomes are included only after a separate result audit.

## Layout

- `CE6190_ICS_Report_20261117.pdf`: exported review draft.
- `paper/main.tex`, `paper/preamble.tex`, `paper/body.tex`, `paper/sections/`: active LaTeX sources.
- `paper/figures/`: four referenced PDF figures, editable SVGs where available,
  previews, figure-generation audits, and exact method-diagram raster inputs.
- `literature/`: bibliography and supporting reference records.
- `audit/`: current reference/use and style audits.
- `logic/`: earlier Chinese paper-to-sentence logic map for historical content review.
- `review/front_logic_ZH.md`: current introduction/method positioning logic review.
- `review/batch2_integration_checklist.md`: conditions for integrating second-batch results.
- `math/`: computed analysis and the original recomputation script.
- `evidence/`: exact six-case qualitative inputs and the component-effect summary,
  sufficient to regenerate the report figures. This is not an experiment backup.
- `SOURCE_INVENTORY.json`: hashes and sizes of this release's included inputs.

Inactive draft sections, the obsolete combined method diagram, datasets, weights,
complete episode ledger, and TeX caches are not included.

## Compile with supplied figures

Use TeX Live or MacTeX. From the repository root:

```bash
cd report/paper
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

This produces `report/paper/main.pdf`. It needs no network, model execution,
raw episode ledger, or figure regeneration. The four supplied PDFs are sufficient.

For automatic compilation, citation checks, page-size checks and all-page previews,
install the packages in `requirements-report.txt` plus Poppler, then run:

```bash
python3 report/paper/build_report.py
```

This writes `report/deliverables/CE6190_ICS_Report_20261117.pdf`,
`report/build/build_validation.json`, and page previews in `report/rendered/`.
The root PDF is the packaged review draft, not automatically overwritten by rebuilding.
Review the rebuilt PDF before replacing the draft. On minimal TeX installations,
the build helper creates local format/font caches under `report/build/`; it does
not change system TeX configuration.

## Regenerate every figure

Install `numpy`, `matplotlib`, and `Pillow` from `requirements-report.txt`, then run:

```bash
python3 report/paper/build_figures.py
```

The packaged version defaults to `report/evidence/`. It regenerates both method
schematics and both result figures using only included inputs. It validates all
twelve displayed predictions' pixel counts and IoUs against the exported records.
No model or collector runs. `--methods-only` and `--results-only` select subsets;
`--evidence-root` can point to an existing larger audited evidence workspace.

Method feature grids and module icons are schematics, while reference/query
images and displayed masks are exact exported evidence. Figure provenance stays
with the sources. Rebuilding figures can change PDF metadata; compare rendered
content as well as hashes when reviewing a regenerated file.

## Recompute full mathematical analysis

The mathematical summaries are included, but the complete audited episode ledger
is intentionally not committed. To recompute rather than merely compile:

```bash
python3 report/math/analyze_mathematical_structure.py \
  --db /path/to/full_analysis/episodes.sqlite \
  --out /path/to/new-math-output
```

The database's parent directory must also contain the existing
`components_analysis.json` and `parameter_cost_size.json`. The script opens SQLite
read-only and uses NumPy. These observed first-batch interactions and exact metric
identities do not establish AGS's second-batch performance.

## Source inventory

`SOURCE_INVENTORY.json` records the release inputs before that inventory file was
added. Its hashes describe the packaged revision, so deliberate edits or figure
regeneration will change them. Keep the original ZIP for an exact release snapshot.

The installer preserves files outside its explicit `report/` payload and makes no
Git commits, pushes, model submissions, or server changes.
