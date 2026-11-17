# Submitted Group report style review — 17 November 2026

## Evidence examined

- Submitted Group PDF: `upload/group_07_report_project2(2).pdf` (14 pages), rendered pages 1, 2, 12, 14.
- Prior Individual PDF: `report_polish_20261117/build/main.pdf` (12 pages), rendered pages 1, 2, 12; PDF text/font positions inspected.
- Visual comparison images are under `audit/style/` (working review files, not required in the repository).

## Matching features retained

Both documents use US Letter pages, Times-family body text at approximately 10 pt, approximately 58.5 pt side margins, two columns with an 18 pt gap, 12 pt section headings, 11 pt subsection headings, 9 pt captions, italic Abstract text, and restrained colored links. The Individual figures already use the Group report’s visual language: pale blue region panels, thin grey arrows, actual image/mask examples, compact serif labels, and small color accents. No scientific figure was redrawn.

The Group title is approximately 14.4 pt bold; the previous Individual title was 15 pt. The previous Individual title started at y=65.8 pt, whereas the submitted Group title starts at y=105.9 pt. This made the Individual front matter look compressed. The revised masthead uses 14.4 pt, a single title line, the Group top position, and spacing adapted to one author and one title line. Final compilation must check that it does not introduce an undesirable page break.

## Applied formatting edits

1. `paper/preamble.tex`: title size 14.4 pt; first-page top/author/Abstract spacing adjusted toward Group style. Subsection numbers now end with a period, as in the submitted Group PDF.
2. Author block retains `HAN SHUANGYUE`, `G2690005A`, and a clickable envelope for `hans0053@e.ntu.edu.sg`. The envelope is reduced to 7×5 pt to match the visual scale of the Group icon. The Group-only badge and equal-contribution markers are not applicable to an individual paper.
3. `paper/sections/introduction.tex`: appended the clickable repository URL to the Abstract, matching the Group report placement. `Our contributions are threefold:` now ends the preceding Introduction paragraph.
4. `paper/sections/appendix.tex`: added an explicit unnumbered `Appendix` heading before section A, preserving appendix labels and the full-width research-matrix table.

No numerical result, method claim, contribution wording, or scientific figure changed in this formatting pass. `main.tex` did not require editing. Bibliographic and scientific usage corrections are handled by the separate audit.

## Final layout checks required

- Compile and render all pages; verify title remains one line without margin overrun.
- Confirm the repository URL wraps at safe points and remains clickable.
- Confirm author email annotation and matriculation number remain present.
- Inspect Introduction end, Figure 1/2 placement, reference-page balance, and Appendix opening after the title-spacing changes.
- Ensure no unresolved references or citations and no clipped diagrams, captions, or tables.

## Final rendered-page review

Final `build/main.pdf` pages 2–7 were inspected at the generated 120 dpi raster size after the bibliography and prose corrections. Verdict: **PASS — no blocking layout defect**.

- Page 2: Figure 1 stays inside the text area; arrows, labels, actual images and masks are legible. The diagram palette/font scale matches the submitted Group report. Related Work starts cleanly below its concise caption.
- Page 3: problem, debiasing and scoring equations are legible; the long pooled-IoU equation keeps its number on a separate baseline without clipping or entering the adjacent column. Section headings and body maintain the intended hierarchy.
- Page 4: Figure 2 has balanced visual panels, clear two-branch flow and a readable caption. The Results text stays within both column margins.
- Page 5: Tables 1–2 align their numeric columns; confidence intervals and latency entries are readable. The interaction equation fits without collision.
- Page 6: Tables 3–4, the AGS decision rule and cost identities all fit. The long Section 5 title uses two lines, with `(AGS)` on the second; this is acceptable but can be shortened editorially in a future content pass.
- Page 7: Figure 3 and Table 5 have consistent label sizes, restrained colors and no clipping; the continuation of the mathematical analysis and evaluation plan is readable. The full-width floats appear after their first text discussion, as expected for two-column LaTeX.

No additional edits were made during this review. Root separately reviews pages 1 and 8–12, including author/repo links and Appendix opening.
